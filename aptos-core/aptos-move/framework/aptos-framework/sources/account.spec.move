spec aptos_framework::account {
    spec module {
        pragma verify = true;
        pragma aborts_if_is_strict;
    }

    /// Convert address to singer and return.
    spec create_signer(addr: address): signer {
        pragma opaque;
        aborts_if [abstract] false;
        ensures [abstract] signer::address_of(result) == addr;
    }

    /// Only the address `@aptos_framework` can call.
    /// OriginatingAddress does not exist under `@aptos_framework` before the call.
    spec initialize(aptos_framework: &signer) {
        let aptos_addr = signer::address_of(aptos_framework);
        aborts_if !system_addresses::is_aptos_framework_address(aptos_addr);
        aborts_if exists<OriginatingAddress>(aptos_addr);
        ensures exists<OriginatingAddress>(aptos_addr);
    }

    /// Check if the bytes of the new address is 32.
    /// The Account does not exist under the new address before creating the account.
    /// Limit the new account address is not @vm_reserved / @aptos_framework / @aptos_toke.
    spec create_account(new_address: address): signer {
        include CreateAccount {addr: new_address};
        aborts_if new_address == @vm_reserved || new_address == @aptos_framework || new_address == @aptos_token;
        ensures signer::address_of(result) == new_address;
    }

    /// Check if the bytes of the new address is 32.
    /// The Account does not exist under the new address before creating the account.
    spec create_account_unchecked(new_address: address): signer {
        include CreateAccount {addr: new_address};
        ensures signer::address_of(result) == new_address;
    }

    spec schema CreateAccount {
        addr: address;
        let authentication_key = bcs::to_bytes(addr);
        aborts_if len(authentication_key) != 32;
        aborts_if exists<Account>(addr);
    }

    spec get_guid_next_creation_num(addr: address): u64 {
        aborts_if !exists<Account>(addr);
        ensures result == global<Account>(addr).guid_creation_num;
    }

    spec get_sequence_number(addr: address): u64 {
        aborts_if !exists<Account>(addr);
        ensures result == global<Account>(addr).sequence_number;
    }

    /// The Account existed under the address.
    /// The sequence_number of the Account is up to MAX_U64.
    spec increment_sequence_number(addr: address) {
        let sequence_number = global<Account>(addr).sequence_number;
        aborts_if !exists<Account>(addr);
        aborts_if sequence_number == MAX_U64;
        modifies global<Account>(addr);
        let post post_sequence_number = global<Account>(addr).sequence_number;
        ensures post_sequence_number == sequence_number + 1;
    }

    spec get_authentication_key(addr: address): vector<u8> {
        aborts_if !exists<Account>(addr);
        ensures result == global<Account>(addr).authentication_key;
    }

    /// The Account existed under the signer before the call.
    /// The length of new_auth_key is 32.
    spec rotate_authentication_key_internal(account: &signer, new_auth_key: vector<u8>) {
        let addr = signer::address_of(account);
        let post account_resource = global<Account>(addr);
        aborts_if !exists<Account>(addr);
        aborts_if vector::length(new_auth_key) != 32;
        modifies global<Account>(addr);
        ensures account_resource.authentication_key == new_auth_key;
    }

    spec assert_valid_rotation_proof_signature_and_get_auth_key(scheme: u8, public_key_bytes: vector<u8>, signature: vector<u8>, challenge: &RotationProofChallenge): vector<u8> {
        include scheme == ED25519_SCHEME ==> ed25519::NewUnvalidatedPublicKeyFromBytesAbortsIf { bytes: public_key_bytes };
        include scheme == ED25519_SCHEME ==> ed25519::NewSignatureFromBytesAbortsIf { bytes: signature };
        aborts_if scheme == ED25519_SCHEME && !ed25519::spec_signature_verify_strict_t(
            ed25519::Signature { bytes: signature },
            ed25519::UnvalidatedPublicKey { bytes: public_key_bytes },
            challenge
        );

        include scheme == MULTI_ED25519_SCHEME ==> multi_ed25519::NewUnvalidatedPublicKeyFromBytesAbortsIf { bytes: public_key_bytes };
        include scheme == MULTI_ED25519_SCHEME ==> multi_ed25519::NewSignatureFromBytesAbortsIf { bytes: signature };
        aborts_if scheme == MULTI_ED25519_SCHEME && !multi_ed25519::spec_signature_verify_strict_t(
            multi_ed25519::Signature { bytes: signature },
            multi_ed25519::UnvalidatedPublicKey { bytes: public_key_bytes },
            challenge
        );
        aborts_if scheme != ED25519_SCHEME && scheme != MULTI_ED25519_SCHEME;
    }

    /// The Account existed under the signer
    /// The authentication scheme is ED25519_SCHEME and MULTI_ED25519_SCHEME
    spec rotate_authentication_key(
        account: &signer,
        from_scheme: u8,
        from_public_key_bytes: vector<u8>,
        to_scheme: u8,
        to_public_key_bytes: vector<u8>,
        cap_rotate_key: vector<u8>,
        cap_update_table: vector<u8>,
    ) {
        // TODO: complex aborts conditions.
        pragma aborts_if_is_partial;
        let addr = signer::address_of(account);
        let account_resource = global<Account>(addr);
        aborts_if !exists<Account>(addr);
        aborts_if from_scheme != ED25519_SCHEME && from_scheme != MULTI_ED25519_SCHEME;
        modifies global<Account>(addr);
        modifies global<OriginatingAddress>(@aptos_framework);
    }

    /// The Account existed under the signer.
    /// The authentication scheme is ED25519_SCHEME and MULTI_ED25519_SCHEME.
    spec offer_signer_capability(
        account: &signer,
        signer_capability_sig_bytes: vector<u8>,
        account_scheme: u8,
        account_public_key_bytes: vector<u8>,
        recipient_address: address
    ) {
        let source_address = signer::address_of(account);
        let account_resource = global<Account>(source_address);
        let proof_challenge = SignerCapabilityOfferProofChallengeV2 {
            sequence_number: account_resource.sequence_number,
            source_address,
            recipient_address,
        };

        aborts_if !exists<Account>(recipient_address);
        aborts_if !exists<Account>(source_address);

        include account_scheme == ED25519_SCHEME ==> ed25519::NewUnvalidatedPublicKeyFromBytesAbortsIf { bytes: account_public_key_bytes };
        aborts_if account_scheme == ED25519_SCHEME && ({
            let expected_auth_key = ed25519::spec_public_key_bytes_to_authentication_key(account_public_key_bytes);
            account_resource.authentication_key != expected_auth_key
        });
        include account_scheme == ED25519_SCHEME ==> ed25519::NewSignatureFromBytesAbortsIf { bytes: signer_capability_sig_bytes };
        aborts_if account_scheme == ED25519_SCHEME && !ed25519::spec_signature_verify_strict_t(
            ed25519::Signature { bytes: signer_capability_sig_bytes },
            ed25519::UnvalidatedPublicKey { bytes: account_public_key_bytes },
            proof_challenge
        );

        include account_scheme == MULTI_ED25519_SCHEME ==> multi_ed25519::NewUnvalidatedPublicKeyFromBytesAbortsIf { bytes: account_public_key_bytes };
        aborts_if account_scheme == MULTI_ED25519_SCHEME && ({
            let expected_auth_key = multi_ed25519::spec_public_key_bytes_to_authentication_key(account_public_key_bytes);
            account_resource.authentication_key != expected_auth_key
        });
        include account_scheme == MULTI_ED25519_SCHEME ==> multi_ed25519::NewSignatureFromBytesAbortsIf { bytes: signer_capability_sig_bytes };
        aborts_if account_scheme == MULTI_ED25519_SCHEME && !multi_ed25519::spec_signature_verify_strict_t(
            multi_ed25519::Signature { bytes: signer_capability_sig_bytes },
            multi_ed25519::UnvalidatedPublicKey { bytes: account_public_key_bytes },
            proof_challenge
        );

        aborts_if account_scheme != ED25519_SCHEME && account_scheme != MULTI_ED25519_SCHEME;

        modifies global<Account>(source_address);
    }

    /// The Account existed under the signer.
    /// The value of signer_capability_offer.for of Account resource under the signer is to_be_revoked_address.
    spec revoke_signer_capability(account: &signer, to_be_revoked_address: address) {
        aborts_if !exists<Account>(to_be_revoked_address);
        let addr = signer::address_of(account);
        let account_resource = global<Account>(addr);
        aborts_if !exists<Account>(addr);
        aborts_if !option::spec_contains(account_resource.signer_capability_offer.for,to_be_revoked_address);
        modifies global<Account>(addr);
        ensures exists<Account>(to_be_revoked_address);
    }

    /// The Account existed under the signer.
    /// The value of signer_capability_offer.for of Account resource under the signer is offerer_address.
    spec create_authorized_signer(account: &signer, offerer_address: address): signer {
        include AccountContainsAddr{
            account: account,
            address: offerer_address,
        };
        modifies global<Account>(offerer_address);
        ensures exists<Account>(offerer_address);
        ensures signer::address_of(result) == offerer_address;
    }

    spec schema AccountContainsAddr {
        account: signer;
        address: address;
        let addr = signer::address_of(account);
        let account_resource = global<Account>(address);
        aborts_if !exists<Account>(address);
        aborts_if !option::spec_contains(account_resource.signer_capability_offer.for,addr);
    }

    /// The Account existed under the signer
    /// The value of signer_capability_offer.for of Account resource under the signer is to_be_revoked_address
    spec create_resource_address(source: &address, seed: vector<u8>): address {
        pragma opaque;
        pragma aborts_if_is_strict = false;
        // This function should not abort assuming the result of `sha3_256` is deserializable into an address.
        aborts_if [abstract] false;
        ensures [abstract] result == spec_create_resource_address(source, seed);
    }

    spec fun spec_create_resource_address(source: address, seed: vector<u8>): address;

    spec create_resource_account(source: &signer, seed: vector<u8>): (signer, SignerCapability) {
        let source_addr = signer::address_of(source);
        let resource_addr = spec_create_resource_address(source_addr, seed);

        aborts_if len(ZERO_AUTH_KEY) != 32;
        include exists_at(resource_addr) ==> CreateResourceAccountAbortsIf;
        include !exists_at(resource_addr) ==> CreateAccount {addr: resource_addr};
    }

    /// Check if the bytes of the new address is 32.
    /// The Account does not exist under the new address before creating the account.
    /// The system reserved addresses is @0x1 / @0x2 / @0x3 / @0x4 / @0x5  / @0x6 / @0x7 / @0x8 / @0x9 / @0xa.
    spec create_framework_reserved_account(addr: address): (signer, SignerCapability) {
        aborts_if spec_is_framework_address(addr);
        include CreateAccount {addr};
        ensures signer::address_of(result_1) == addr;
        ensures result_2 == SignerCapability { account: addr };
    }

    spec fun spec_is_framework_address(addr: address): bool{
        addr != @0x1 &&
        addr != @0x2 &&
        addr != @0x3 &&
        addr != @0x4 &&
        addr != @0x5 &&
        addr != @0x6 &&
        addr != @0x7 &&
        addr != @0x8 &&
        addr != @0x9 &&
        addr != @0xa
    }

    /// The Account existed under the signer.
    /// The guid_creation_num of the ccount resource is up to MAX_U64.
    spec create_guid(account_signer: &signer): guid::GUID {
        let addr = signer::address_of(account_signer);
        let account = global<Account>(addr);
        aborts_if !exists<Account>(addr);
        aborts_if account.guid_creation_num + 1 > MAX_U64;
        modifies global<Account>(addr);
    }

    /// The Account existed under the signer.
    /// The guid_creation_num of the Account is up to MAX_U64.
    spec new_event_handle<T: drop + store>(account: &signer): EventHandle<T> {
        let addr = signer::address_of(account);
        let account = global<Account>(addr);
        aborts_if !exists<Account>(addr);
        aborts_if account.guid_creation_num + 1 > MAX_U64;
    }

    spec register_coin<CoinType>(account_addr: address) {
        aborts_if !exists<Account>(account_addr);
        aborts_if !type_info::spec_is_struct<CoinType>();
        modifies global<Account>(account_addr);
    }

    spec create_signer_with_capability(capability: &SignerCapability): signer {
        let addr = capability.account;
        ensures signer::address_of(result) == addr;
    }

    spec schema CreateResourceAccountAbortsIf {
        resource_addr: address;
        let account = global<Account>(resource_addr);
        aborts_if len(account.signer_capability_offer.for.vec) != 0;
        aborts_if account.sequence_number != 0;
    }
}
