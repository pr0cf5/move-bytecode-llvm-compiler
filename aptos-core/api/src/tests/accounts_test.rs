// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use super::new_test_context;
use aptos_api_test_context::{current_function_name, find_value};
use aptos_api_types::{MoveModuleBytecode, MoveResource, StateKeyWrapper};
use serde_json::json;
use std::str::FromStr;

/* TODO: reactivate once cause of failure for `"8"` vs `8` in the JSON output is known.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_account_resources_returns_empty_array_for_account_has_no_resources() {
    let mut context = new_test_context(current_function_name!());
    let address = "0x1";

    let resp = context.get(&account_resources(address)).await;
    context.check_golden_output(resp);
}
 */

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_account_resources_by_address_0x0() {
    let mut context = new_test_context(current_function_name!());
    let address = "0x0";

    let resp = context
        .expect_status_code(404)
        .get(&account_resources(address))
        .await;
    context.check_golden_output(resp);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_account_resources_by_invalid_address_missing_0x_prefix() {
    let mut context = new_test_context(current_function_name!());
    let invalid_addresses = vec!["1", "0xzz", "01"];
    for invalid_address in &invalid_addresses {
        let resp = context
            .expect_status_code(400)
            .get(&account_resources(invalid_address))
            .await;
        context.check_golden_output(resp);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_account_resources_by_valid_account_address() {
    let context = new_test_context(current_function_name!());
    let addresses = vec!["0x1", "0x00000000000000000000000000000001"];
    for address in &addresses {
        context.get(&account_resources(address)).await;
    }
}

// Unstable due to framework changes
#[ignore]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_account_resources_response() {
    let mut context = new_test_context(current_function_name!());
    let address = "0x1";

    let resp = context.get(&account_resources(address)).await;
    context.check_golden_output(resp);
}

// Unstable due to framework changes
#[ignore]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_account_modules() {
    let mut context = new_test_context(current_function_name!());
    let address = "0x1";

    let resp = context.get(&account_modules(address)).await;
    context.check_golden_output(resp);
}

// Unstable due to framework changes
#[ignore]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_module_with_entry_functions() {
    let mut context = new_test_context(current_function_name!());
    let address = "0x1";

    let resp = context.get(&account_modules(address)).await;
    context.check_golden_output(resp);
}

// Unstable due to framework changes
#[ignore]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_module_aptos_config() {
    let mut context = new_test_context(current_function_name!());
    let address = "0x1";

    let resp = context.get(&account_modules(address)).await;
    context.check_golden_output(resp);
}

// Unstable due to framework changes
#[ignore]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_account_modules_structs() {
    let mut context = new_test_context(current_function_name!());
    let address = "0x1";

    let resp = context.get(&account_modules(address)).await;
    context.check_golden_output(resp);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_account_resources_by_ledger_version() {
    let mut context = new_test_context(current_function_name!());
    let account = context.gen_account();
    let txn = context.create_user_account(&account);
    context.commit_block(&vec![txn.clone()]).await;

    let ledger_version_1_resources = context
        .get(&account_resources(
            &context.root_account().address().to_hex_literal(),
        ))
        .await;
    let root_account = find_value(&ledger_version_1_resources, |f| {
        f["type"] == "0x1::account::Account"
    });
    assert_eq!(root_account["data"]["sequence_number"], "1");

    let ledger_version_0_resources = context
        .get(&account_resources_with_ledger_version(
            &context.root_account().address().to_hex_literal(),
            0,
        ))
        .await;
    let root_account = find_value(&ledger_version_0_resources, |f| {
        f["type"] == "0x1::account::Account"
    });
    assert_eq!(root_account["data"]["sequence_number"], "0");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_account_resources_by_too_large_ledger_version() {
    let mut context = new_test_context(current_function_name!());
    let resp = context
        .expect_status_code(404)
        .get(&account_resources_with_ledger_version(
            &context.root_account().address().to_hex_literal(),
            1000000000000000000,
        ))
        .await;
    context.check_golden_output(resp);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_account_resources_by_invalid_ledger_version() {
    let mut context = new_test_context(current_function_name!());
    let resp = context
        .expect_status_code(400)
        .get(&account_resources_with_ledger_version(
            &context.root_account().address().to_hex_literal(),
            -1,
        ))
        .await;
    context.check_golden_output(resp);
}

// figure out a working module code, no idea where the existing one comes from
#[ignore] // TODO(issue 81): re-enable after cleaning up the compiled code in the test
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_account_modules_by_ledger_version() {
    let mut context = new_test_context(current_function_name!());
    let code = "a11ceb0b0300000006010002030205050703070a0c0816100c260900000001000100000102084d794d6f64756c650269640000000000000000000000000b1e55ed00010000000231010200";
    let mut root_account = context.root_account();
    let txn = root_account.sign_with_transaction_builder(
        context
            .transaction_factory()
            .module(hex::decode(code).unwrap()),
    );
    context.commit_block(&vec![txn.clone()]).await;
    let modules = context
        .get(&account_modules(
            &context.root_account().address().to_hex_literal(),
        ))
        .await;

    assert_ne!(modules, json!([]));

    let modules = context
        .get(&account_modules_with_ledger_version(
            &context.root_account().address().to_hex_literal(),
            0,
        ))
        .await;
    assert_eq!(modules, json!([]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_core_account_data() {
    let mut context = new_test_context(current_function_name!());
    let resp = context.get("/accounts/0x1").await;
    context.check_golden_output(resp);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_core_account_data_not_found() {
    let mut context = new_test_context(current_function_name!());
    let resp = context.expect_status_code(404).get("/accounts/0xf").await;
    context.check_golden_output(resp);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_account_resources_with_pagination() {
    let context = new_test_context(current_function_name!());
    let address = "0x1";

    // Make a request with no limit. We'll use this full list of resources
    // as a comparison with the results from using pagination parameters.
    // There should be no cursor in the header in this case. Note: This won't
    // be true if for some reason the account used in this test has more than
    // the default max page size for resources (1000 at the time of writing,
    // based on config/src/config/api_config.rs).
    let req = warp::test::request()
        .method("GET")
        .path(&format!("/v1{}", account_resources(address)));
    let resp = context.reply(req).await;
    assert_eq!(resp.status(), 200);
    assert!(!resp.headers().contains_key("X-Aptos-Cursor"));
    let all_resources: Vec<MoveResource> = serde_json::from_slice(resp.body()).unwrap();
    // We assert there are at least 10 resources. If there aren't, the rest of the
    // test will be wrong.
    assert!(all_resources.len() >= 10);

    // Make a request, assert we get a cursor back in the header for the next
    // page of results. Assert we can deserialize the string representation
    // of the cursor returned in the header.
    let req = warp::test::request()
        .method("GET")
        .path(&format!("/v1{}?limit=5", account_resources(address)));
    let resp = context.reply(req).await;
    assert_eq!(resp.status(), 200);
    let cursor_header = resp
        .headers()
        .get("X-Aptos-Cursor")
        .expect("Cursor header was missing");
    let cursor_header = StateKeyWrapper::from_str(cursor_header.to_str().unwrap()).unwrap();
    let resources: Vec<MoveResource> = serde_json::from_slice(resp.body()).unwrap();
    assert_eq!(resources.len(), 5);
    assert_eq!(resources, all_resources[0..5].to_vec());

    // Make a request using the cursor. Assert the 5 results we get back are the next 5.
    let req = warp::test::request().method("GET").path(&format!(
        "/v1{}?limit=5&start={}",
        account_resources(address),
        cursor_header
    ));
    let resp = context.reply(req).await;
    assert_eq!(resp.status(), 200);
    let cursor_header = resp
        .headers()
        .get("X-Aptos-Cursor")
        .expect("Cursor header was missing");
    let cursor_header = StateKeyWrapper::from_str(cursor_header.to_str().unwrap()).unwrap();
    let resources: Vec<MoveResource> = serde_json::from_slice(resp.body()).unwrap();
    assert_eq!(resources.len(), 5);
    assert_eq!(resources, all_resources[5..10].to_vec());

    // Get the rest of the resources, assert there is no cursor now.
    let req = warp::test::request().method("GET").path(&format!(
        "/v1{}?limit=1000&start={}",
        account_resources(address),
        cursor_header
    ));
    let resp = context.reply(req).await;
    assert_eq!(resp.status(), 200);
    assert!(!resp.headers().contains_key("X-Aptos-Cursor"));
    let resources: Vec<MoveResource> = serde_json::from_slice(resp.body()).unwrap();
    assert_eq!(resources.len(), all_resources.len() - 10);
    assert_eq!(resources, all_resources[10..].to_vec());
}

// Same as the above test but for modules.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_account_modules_with_pagination() {
    let context = new_test_context(current_function_name!());
    let address = "0x1";

    // Make a request with no limit. We'll use this full list of modules
    // as a comparison with the results from using pagination parameters.
    // There should be no cursor in the header in this case. Note: This won't
    // be true if for some reason the account used in this test has more than
    // the default max page size for modules (1000 at the time of writing,
    // based on config/src/config/api_config.rs).
    let req = warp::test::request()
        .method("GET")
        .path(&format!("/v1{}", account_modules(address)));
    let resp = context.reply(req).await;
    assert_eq!(resp.status(), 200);
    assert!(!resp.headers().contains_key("X-Aptos-Cursor"));
    let all_modules: Vec<MoveModuleBytecode> = serde_json::from_slice(resp.body()).unwrap();
    // We assert there are at least 10 modules. If there aren't, the rest of the
    // test will be wrong.
    assert!(all_modules.len() >= 10);

    // Make a request, assert we get a cursor back in the header for the next
    // page of results. Assert we can deserialize the string representation
    // of the cursor returned in the header.
    let req = warp::test::request()
        .method("GET")
        .path(&format!("/v1{}?limit=5", account_modules(address)));
    let resp = context.reply(req).await;
    assert_eq!(resp.status(), 200);
    let cursor_header = resp
        .headers()
        .get("X-Aptos-Cursor")
        .expect("Cursor header was missing");
    let cursor_header = StateKeyWrapper::from_str(cursor_header.to_str().unwrap()).unwrap();
    let modules: Vec<MoveModuleBytecode> = serde_json::from_slice(resp.body()).unwrap();
    assert_eq!(modules.len(), 5);
    assert_eq!(modules, all_modules[0..5].to_vec());

    // Make a request using the cursor. Assert the 5 results we get back are the next 5.
    let req = warp::test::request().method("GET").path(&format!(
        "/v1{}?limit=5&start={}",
        account_modules(address),
        cursor_header
    ));
    let resp = context.reply(req).await;
    assert_eq!(resp.status(), 200);
    let cursor_header = resp
        .headers()
        .get("X-Aptos-Cursor")
        .expect("Cursor header was missing");
    let cursor_header = StateKeyWrapper::from_str(cursor_header.to_str().unwrap()).unwrap();
    let modules: Vec<MoveModuleBytecode> = serde_json::from_slice(resp.body()).unwrap();
    assert_eq!(modules.len(), 5);
    assert_eq!(modules, all_modules[5..10].to_vec());

    // Get the rest of the modules, assert there is no cursor now.
    let req = warp::test::request().method("GET").path(&format!(
        "/v1{}?limit=1000&start={}",
        account_modules(address),
        cursor_header
    ));
    let resp = context.reply(req).await;
    assert_eq!(resp.status(), 200);
    assert!(!resp.headers().contains_key("X-Aptos-Cursor"));
    let modules: Vec<MoveModuleBytecode> = serde_json::from_slice(resp.body()).unwrap();
    assert_eq!(modules.len(), all_modules.len() - 10);
    assert_eq!(modules, all_modules[10..].to_vec());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_account_items_limit_params() {
    let context = new_test_context(current_function_name!());
    let address = "0x1";

    // Ensure limit=0 is rejected.
    let req = warp::test::request()
        .method("GET")
        .path(&format!("/v1{}?limit=0", account_resources(address)));
    let resp = context.reply(req).await;
    assert_eq!(resp.status(), 400);

    // Ensure limit=0 is rejected.
    let req = warp::test::request()
        .method("GET")
        .path(&format!("/v1{}?limit=0", account_modules(address)));
    let resp = context.reply(req).await;
    assert_eq!(resp.status(), 400);

    // Ensure garbage start param values are rejected.
    let req = warp::test::request().method("GET").path(&format!(
        "/v1{}?start=iwouldnotsurviveavibecheckrightnow",
        account_modules(address)
    ));
    let resp = context.reply(req).await;
    assert_eq!(resp.status(), 400);
}

fn account_resources(address: &str) -> String {
    format!("/accounts/{}/resources", address)
}

fn account_resources_with_ledger_version(address: &str, ledger_version: i128) -> String {
    format!(
        "{}?ledger_version={}",
        account_resources(address),
        ledger_version
    )
}

fn account_modules(address: &str) -> String {
    format!("/accounts/{}/modules", address)
}

fn account_modules_with_ledger_version(address: &str, ledger_version: i128) -> String {
    format!(
        "{}?ledger_version={}",
        account_modules(address),
        ledger_version
    )
}
