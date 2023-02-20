spec aptos_framework::aggregator_factory {
    spec module {
        pragma verify = true;
        pragma aborts_if_is_strict;
    }

    spec new_aggregator(aggregator_factory: &mut AggregatorFactory, limit: u128): Aggregator {
        pragma opaque;
        aborts_if false;
        ensures result.limit == limit;
    }

    /// Make sure the caller is @aptos_framework.
    /// AggregatorFactory is not under the caller before creating the resource.
    spec initialize_aggregator_factory(aptos_framework: &signer) {
        use std::signer;
        let addr = signer::address_of(aptos_framework);
        aborts_if addr != @aptos_framework;
        aborts_if exists<AggregatorFactory>(addr);
        ensures exists<AggregatorFactory>(addr);
    }

    spec create_aggregator_internal(limit: u128): Aggregator {
        aborts_if !exists<AggregatorFactory>(@aptos_framework);
    }

    /// Make sure the caller is @aptos_framework.
    /// AggregatorFactory existed under the @aptos_framework when Creating a new aggregator.
    spec create_aggregator(account: &signer, limit: u128): Aggregator {
        use std::signer;
        let addr = signer::address_of(account);
        aborts_if addr != @aptos_framework;
        aborts_if !exists<AggregatorFactory>(@aptos_framework);
    }
}
