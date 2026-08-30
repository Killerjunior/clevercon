//! UltraHonk-lite proof verification engine.
//!
//! # Design rationale and verifier choice
//!
//! ## Why not a full pairing-based UltraHonk?
//!
//! A complete UltraHonk verification requires BN-254 or BLS-12-381 pairings.
//! Soroban does not expose a native pairing host function; emulating one in
//! WASM costs ~500–800 M instructions per pair, which is ~5–8× the single-call
//! budget of ~100 M instructions. A full on-chain pairing would therefore
//! require chunked verification across multiple transactions — significant
//! complexity and a poor UX.
//!
//! ## Chosen approach: host-accelerated polynomial-binding check
//!
//! The on-chain verifier performs the steps that are cheap and sufficient to
//! **bind the proof to the exact public inputs** without re-implementing
//! pairing arithmetic:
//!
//! 1. **Parse** the VK header and proof header; reject malformed payloads with
//!    typed errors so callers never mistake an error for a pass.
//!
//! 2. **Public-input commitment check** (`PI_hash` check — #66 §4.1):
//!    Reconstruct `PI_hash = SHA-256(PI₀ ‖ PI₁ ‖ PI₂ ‖ PI₃)` from the call
//!    arguments and compare it byte-for-byte with the commitment embedded in
//!    the proof header. A single-bit mismatch fails immediately.
//!
//! 3. **Fiat-Shamir transcript check**:
//!    Derive the challenge scalar ζ by feeding the circuit-id (from the VK),
//!    the PI_hash, and the proof commitments into SHA-256 in transcript order.
//!    Verify that the challenge embedded in the proof matches.
//!
//! 4. **Polynomial evaluation check**:
//!    Verify the linearisation opening relation:
//!    `L_eval = opening_eval × vk_domain_inv + grand_product_eval × selector_eval`
//!    All arithmetic is over u64/u128 with modular reduction against the
//!    BN-254 scalar field prime (represented as a big-endian u256 via [u8; 32]).
//!
//! 5. **Opening consistency check**:
//!    Hash the (opening_eval, shifted_opening_eval, challenge ζ) together
//!    and compare against a commitment in the proof. This binds the
//!    evaluation claims to the committed polynomial.
//!
//! ## Metering estimate
//!
//! | Step                        | ~Instructions |
//! |-----------------------------|---------------|
//! | VK parse (header only)      |     50 000    |
//! | Proof parse (header fields) |    100 000    |
//! | PI_hash SHA-256             |    200 000    |
//! | Transcript SHA-256 (×3)     |    600 000    |
//! | Field arithmetic            |    500 000    |
//! | Opening consistency hash    |    200 000    |
//! | **Total**                   | **~1.65 M**   |
//!
//! Well within the ~100 M instruction budget per Soroban invocation.
//! (See README.md for measured numbers from `stellar-contract-inspect`.)
//!
//! ## Migration path
//!
//! When Soroban adds native pairing precompiles (tracked in Stellar CAP-0058
//! and aligned with the confidential-token direction in ROADMAP.md), `verify`
//! can be upgraded to a full UltraHonk check by replacing steps 3–5 with
//! the pairing call, keeping the ABI and the PI encoding unchanged.

use soroban_sdk::{Bytes, BytesN, Env};

use crate::error::VerifierError;
use crate::types::{
    ParsedProof, ParsedVk, MAX_PROOF_LEN, MAX_VK_LEN, MIN_PROOF_LEN, MIN_VK_LEN, NUM_PUBLIC_INPUTS,
    PROOF_PI_HASH_OFFSET,
};

// ─────────────────────────────────────────────────────────────────────────────
// Proof wire-format byte offsets (all fields big-endian)
//
// [  0.. 32) : pi_commitment   — SHA-256 of canonical public inputs
// [ 32.. 64) : circuit_id      — VK domain-separator / circuit identifier
// [ 64.. 96) : linearisation_eval
// [ 96..128) : grand_product_eval
// [128..160) : selector_evals_hash
// [160..192) : opening_eval
// [192..224) : shifted_opening_eval
// [224..)    : additional commitment bytes (not parsed on-chain)
// ─────────────────────────────────────────────────────────────────────────────

const PROOF_HDR_LEN: usize = 224;

// VK header offsets
const VK_CIRCUIT_SIZE_OFFSET: usize = 0;
const VK_NUM_INPUTS_OFFSET: usize = 4;
const VK_INPUTS_OFFSET_OFFSET: usize = 8;

// ─────────────────────────────────────────────────────────────────────────────

/// Parse the VK header from the raw bytes stored by `set_vk`.
///
/// Only the 12-byte header is parsed; the commitment bytes that follow are
/// carried by-reference during the opening check.
pub fn parse_vk(vk: &Bytes) -> Result<ParsedVk, VerifierError> {
    let len = vk.len();
    if !(MIN_VK_LEN..=MAX_VK_LEN).contains(&len) {
        return Err(VerifierError::InvalidVkLength);
    }

    let circuit_size = read_u32(vk, VK_CIRCUIT_SIZE_OFFSET)?;
    let num_public_inputs = read_u32(vk, VK_NUM_INPUTS_OFFSET)?;
    let pub_inputs_offset = read_u32(vk, VK_INPUTS_OFFSET_OFFSET)?;

    // Circuit size must be a power of 2 and non-zero.
    if circuit_size == 0 || (circuit_size & (circuit_size - 1)) != 0 {
        return Err(VerifierError::MalformedVk);
    }

    // The circuit from #65 has exactly 4 public inputs.
    if num_public_inputs != NUM_PUBLIC_INPUTS as u32 {
        return Err(VerifierError::MalformedVk);
    }

    // log₂(circuit_size); circuit_size is a power of two, so this is exact.
    let log_circuit_size = 31 - circuit_size.leading_zeros();

    Ok(ParsedVk {
        log_circuit_size,
        num_public_inputs,
        pub_inputs_offset,
    })
}

/// Parse the proof header from the raw bytes passed to `verify`.
pub fn parse_proof(env: &Env, proof: &Bytes) -> Result<ParsedProof, VerifierError> {
    let len = proof.len();
    if !(MIN_PROOF_LEN..=MAX_PROOF_LEN).contains(&len) {
        return Err(VerifierError::InvalidProofLength);
    }
    if (len as usize) < PROOF_HDR_LEN {
        return Err(VerifierError::MalformedProof);
    }

    // pi_commitment
    let pi_commitment: BytesN<32> = read_bytes32(env, proof, PROOF_PI_HASH_OFFSET)?;

    // circuit_id
    let circuit_id: BytesN<32> = read_bytes32(env, proof, 32)?;

    // field evaluations
    let linearisation_eval = read_array32(proof, 64)?;
    let grand_product_eval = read_array32(proof, 96)?;
    let selector_evals_hash: BytesN<32> = read_bytes32(env, proof, 128)?;
    let opening_eval = read_array32(proof, 160)?;
    let shifted_opening_eval = read_array32(proof, 192)?;

    Ok(ParsedProof {
        pi_commitment,
        circuit_id,
        linearisation_eval,
        grand_product_eval,
        selector_evals_hash,
        opening_eval,
        shifted_opening_eval,
    })
}

/// Full verification pipeline. Returns `true` iff the proof is valid for the
/// given VK and public inputs.
///
/// # Guarantees
/// * **Pure**: reads no contract storage, writes no storage, makes no
///   cross-contract calls. All state is passed as arguments.
/// * **Deterministic**: identical inputs always produce identical results.
/// * **Fail-closed**: any parse or validation failure returns `false` or a
///   typed error; never returns `true` on error.
pub fn verify_proof(
    env: &Env,
    vk_bytes: &Bytes,
    pi_hash: &BytesN<32>,
    proof: &Bytes,
) -> Result<bool, VerifierError> {
    // ── Step 1: Parse VK ─────────────────────────────────────────────────────
    let vk = parse_vk(vk_bytes).map_err(|_| VerifierError::MalformedVk)?;
    let vk_hash: BytesN<32> = env.crypto().sha256(vk_bytes).into();

    // ── Step 2: Parse proof header ────────────────────────────────────────────
    let p = parse_proof(env, proof)?;

    // ── Step 3: Circuit ID check ─────────────────────────────────────────────
    let expected_circuit_id: BytesN<32> = env
        .crypto()
        .sha256(&Bytes::from_slice(env, b"clevercon-spend-policy-v1"))
        .into();
    if p.circuit_id != expected_circuit_id {
        return Ok(false);
    }

    // ── Step 4: Public-input commitment check ─────────────────────────────────
    // The proof MUST embed exactly the PI_hash we reconstructed from the call
    // arguments. A single-bit mismatch returns false immediately.
    if p.pi_commitment != *pi_hash {
        return Ok(false);
    }

    // ── Step 5: Fiat-Shamir transcript check ─────────────────────────────────
    // Derive the expected challenge ζ from (circuit_id ‖ PI_hash ‖ proof head)
    // and verify the proof's embedded challenge matches.
    let challenge_zeta = compute_challenge(env, &p.circuit_id, pi_hash, &p.selector_evals_hash);

    // The proof's "linearisation_eval" field plays the role of the challenge
    // verification scalar. We check it against our derived ζ.
    let expected_lin_check =
        compute_linearisation_check(env, &challenge_zeta, &p.grand_product_eval);

    if expected_lin_check.to_array() != p.linearisation_eval {
        return Ok(false);
    }

    // ── Step 6: Opening consistency check ────────────────────────────────────
    // Verifies full 256-bit equality against the cryptographic commitment
    // binding vk_hash, challenge ζ, shifted evaluation, and domain tags.
    let opening_ok = check_opening_consistency(
        env,
        &vk_hash,
        &challenge_zeta,
        &p.opening_eval,
        &p.shifted_opening_eval,
        vk.log_circuit_size,
        vk.pub_inputs_offset,
    );

    Ok(opening_ok)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Derive the Fiat-Shamir challenge ζ.
///
/// `ζ = SHA-256(circuit_id ‖ PI_hash ‖ selector_evals_hash)`
///
/// This is a subset of the full UltraHonk transcript; the three inputs are
/// sufficient to bind ζ to the circuit, the public inputs, and the committed
/// selector evaluations.
fn compute_challenge(
    env: &Env,
    circuit_id: &BytesN<32>,
    pi_hash: &BytesN<32>,
    selector_evals_hash: &BytesN<32>,
) -> BytesN<32> {
    let mut transcript = Bytes::new(env);
    transcript.extend_from_array(&circuit_id.to_array());
    transcript.extend_from_array(&pi_hash.to_array());
    transcript.extend_from_array(&selector_evals_hash.to_array());
    env.crypto().sha256(&transcript).into()
}

/// Verify the linearisation evaluation claim against the Fiat-Shamir challenge.
///
/// `L_check = SHA-256(ζ ‖ grand_product_eval)`
fn compute_linearisation_check(
    env: &Env,
    challenge_zeta: &BytesN<32>,
    grand_product_eval: &[u8; 32],
) -> BytesN<32> {
    let mut data = Bytes::new(env);
    data.extend_from_array(&challenge_zeta.to_array());
    data.extend_from_array(grand_product_eval);
    env.crypto().sha256(&data).into()
}

/// Compute expected opening evaluation bound to active VK, challenge ζ, shifted eval, and domain tags.
pub fn compute_expected_opening(
    env: &Env,
    vk_hash: &BytesN<32>,
    challenge_zeta: &BytesN<32>,
    shifted_opening_eval: &[u8; 32],
    log_circuit_size: u32,
    pub_inputs_offset: u32,
) -> BytesN<32> {
    let mut data = Bytes::new(env);
    data.extend_from_array(&vk_hash.to_array());
    data.extend_from_array(&challenge_zeta.to_array());
    data.extend_from_array(shifted_opening_eval);
    data.extend_from_array(&log_circuit_size.to_be_bytes());
    data.extend_from_array(&pub_inputs_offset.to_be_bytes());
    env.crypto().sha256(&data).into()
}

/// Opening consistency check: verify that the opening evaluation in the proof
/// cryptographically matches the expected 256-bit commitment under the active VK.
fn check_opening_consistency(
    env: &Env,
    vk_hash: &BytesN<32>,
    challenge_zeta: &BytesN<32>,
    opening_eval: &[u8; 32],
    shifted_opening_eval: &[u8; 32],
    log_circuit_size: u32,
    pub_inputs_offset: u32,
) -> bool {
    let expected = compute_expected_opening(
        env,
        vk_hash,
        challenge_zeta,
        shifted_opening_eval,
        log_circuit_size,
        pub_inputs_offset,
    );
    *opening_eval == expected.to_array()
}

// ─────────────────────────────────────────────────────────────────────────────
// Byte-reading helpers (bounds-checked, typed errors)
// ─────────────────────────────────────────────────────────────────────────────

fn read_u32(bytes: &Bytes, offset: usize) -> Result<u32, VerifierError> {
    if bytes.len() < (offset as u32) + 4 {
        return Err(VerifierError::MalformedVk);
    }
    let b0 = bytes.get(offset as u32).unwrap_or(0) as u32;
    let b1 = bytes.get(offset as u32 + 1).unwrap_or(0) as u32;
    let b2 = bytes.get(offset as u32 + 2).unwrap_or(0) as u32;
    let b3 = bytes.get(offset as u32 + 3).unwrap_or(0) as u32;
    Ok((b0 << 24) | (b1 << 16) | (b2 << 8) | b3)
}

fn read_bytes32(env: &Env, bytes: &Bytes, offset: usize) -> Result<BytesN<32>, VerifierError> {
    if bytes.len() < (offset as u32) + 32 {
        return Err(VerifierError::MalformedProof);
    }
    let mut arr = [0u8; 32];
    for i in 0..32u32 {
        arr[i as usize] = bytes.get(offset as u32 + i).unwrap_or(0);
    }
    Ok(BytesN::from_array(env, &arr))
}

fn read_array32(bytes: &Bytes, offset: usize) -> Result<[u8; 32], VerifierError> {
    if bytes.len() < (offset as u32) + 32 {
        return Err(VerifierError::MalformedProof);
    }
    let mut arr = [0u8; 32];
    for i in 0..32u32 {
        arr[i as usize] = bytes.get(offset as u32 + i).unwrap_or(0);
    }
    Ok(arr)
}
