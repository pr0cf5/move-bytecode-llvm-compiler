/// Maintains feature flags.
spec std::features {
    spec Features {
        pragma bv=b"0";
    }

    spec set(features: &mut vector<u8>, feature: u64, include: bool) {
        pragma bv=b"0";
        aborts_if false;
        ensures feature / 8 < len(features);
        ensures include == (((int2bv(((1 as u8) << ((feature % (8 as u64)) as u64) as u8)) as u8)
            & features[feature/8] as u8) > (0 as u8));
    }

    spec contains(features: &vector<u8>, feature: u64): bool {
        pragma bv=b"0";
        aborts_if false;
        ensures result == ((feature / 8) < len(features) && ((int2bv((((1 as u8) << ((feature % (8 as u64)) as u64)) as u8)) as u8)
            & features[feature/8] as u8) > (0 as u8));
    }

    spec change_feature_flags(framework: &signer, enable: vector<u64>, disable: vector<u64>) {
        pragma opaque;
        modifies global<Features>(@std);
        aborts_if signer::address_of(framework) != @std;
    }

    spec is_enabled(feature: u64): bool {
        pragma opaque;
        aborts_if [abstract] false;
        ensures [abstract] result == spec_is_enabled(feature);
    }

    spec fun spec_is_enabled(feature: u64): bool;
}
