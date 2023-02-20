// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use anyhow::{bail, Result};
use move_binary_format::CompiledModule;
use move_core_types::{
    account_address::AccountAddress,
    identifier::{IdentStr, Identifier},
};
use move_ir_compiler::Compiler;
use move_stdlib::natives::{all_natives, GasParameters};
use move_table_extension::{self, NativeTableContext};
use move_vm_runtime::{
    move_vm::MoveVM, native_extensions::NativeContextExtensions, native_functions::NativeFunction,
};
use move_vm_test_utils::InMemoryStorage;
use move_vm_types::{
    gas::UnmeteredGasMeter, loaded_data::runtime_types::Type, natives::function::NativeResult,
    pop_arg, values::Value,
};
use smallvec::smallvec;
use std::{collections::VecDeque, env, fs, sync::Arc};

fn make_native_create_signer() -> NativeFunction {
    Arc::new(|_context, ty_args: Vec<Type>, mut args: VecDeque<Value>| {
        assert!(ty_args.is_empty());
        assert!(args.len() == 1);

        let address = pop_arg!(args, AccountAddress);

        Ok(NativeResult::ok(0.into(), smallvec![Value::signer(
            address
        )]))
    })
}

fn compile_test_modules() -> Vec<CompiledModule> {
    let module_sources = [
        r#"
            module 0x1.Test {
                native public create_signer(addr: address): signer;
            }
        "#,
        r#"
            module 0x1.bcs {
                native public to_bytes<MoveValue>(v: &MoveValue): vector<u8>;
            }
        "#,
        r#"
            module 0x1.hash {
                native public sha2_256(data: vector<u8>): vector<u8>;
                native public sha3_256(data: vector<u8>): vector<u8>;
            }
        "#,
        r#"
            module 0x1.table {
                struct Table<phantom K: copy + drop, phantom V> has store {
                    handle: address,
                    length: u64,
                }

                struct Box<V> has key, drop, store {
                    val: V
                }

                native new_table_handle<K, V>(): address;
                native add_box<K: copy + drop, V, B>(table: &mut Self.Table<K, V>, key: K, val: Self.Box<V>);
                native borrow_box<K: copy + drop, V, B>(table: &Self.Table<K, V>, key: K): &Self.Box<V>;
                native borrow_box_mut<K: copy + drop, V, B>(table: &mut Self.Table<K, V>, key: K): &mut Self.Box<V>;
                native contains_box<K: copy + drop, V, B>(table: &Self.Table<K, V>, key: K): bool;
                native remove_box<K: copy + drop, V, B>(table: &mut Self.Table<K, V>, key: K): Self.Box<V>;
                native destroy_empty_box<K: copy + drop, V, B>(table: &Self.Table<K, V>);
                native drop_unchecked_box<K: copy + drop, V, B>(table: Self.Table<K, V>);

                public new<K: copy + drop, V: store>(): Self.Table<K, V> {
                label b0:
                    return Table<K, V> {
                        handle: Self.new_table_handle<K, V>(),
                        length: 0,
                    };
                }

                public destroy_empty<K: copy + drop, V>(table: Self.Table<K, V>) {
                label b0:
                    Self.destroy_empty_box<K, V, Self.Box<V>>(&table);
                    Self.drop_unchecked_box<K, V, Self.Box<V>>(move(table));
                    return;
                }

                public add<K: copy + drop, V>(table: &mut Self.Table<K, V>, key: K, val: V) {
                    let b: Self.Box<V>;
                label b0:
                    b = Box<V> { val: move(val) };
                    Self.add_box<K, V, Self.Box<V>>(move(table), move(key), move(b));
                    return;
                }

                public borrow<K: copy + drop, V>(table: &Self.Table<K, V>, key: K): &V {
                label b0:
                    return &Self.borrow_box<K, V, Self.Box<V>>(move(table), move(key)).Box<V>::val;
                }

                public contains<K: copy + drop, V>(table: &Self.Table<K, V>, key: K): bool {
                label b0:
                    return Self.contains_box<K, V, Self.Box<V>>(move(table), move(key));
                }

                public remove<K: copy + drop, V>(table: &mut Self.Table<K, V>, key: K): V {
                    let v: V;
                label b0:
                    Box<V> { v } = Self.remove_box<K, V, Self.Box<V>>(move(table), move(key));
                    return move(v);
                }
            }
        "#,
    ];

    module_sources
        .into_iter()
        .map(|src| Compiler::new(vec![]).into_compiled_module(src).unwrap())
        .collect()
}

fn main() -> Result<()> {
    let args = env::args().collect::<Vec<_>>();

    if args.len() != 2 {
        bail!("Wrong number of arguments.")
    }

    let stdlib_addr = AccountAddress::from_hex_literal("0x1").unwrap();
    let mut natives = all_natives(stdlib_addr, GasParameters::zeros());
    natives.push((
        stdlib_addr,
        Identifier::new("Test").unwrap(),
        Identifier::new("create_signer").unwrap(),
        make_native_create_signer(),
    ));
    natives.extend(move_table_extension::table_natives(
        stdlib_addr,
        move_table_extension::GasParameters::zeros(),
    ));

    let vm = MoveVM::new(natives).unwrap();
    let mut storage = InMemoryStorage::new();

    let test_modules = compile_test_modules();
    for module in &test_modules {
        let mut blob = vec![];
        module.serialize(&mut blob).unwrap();
        storage.publish_or_overwrite_module(module.self_id(), blob);
    }

    let mut extensions = NativeContextExtensions::default();
    extensions.add(NativeTableContext::new([0; 32], &storage));

    let mut sess = vm.new_session_with_extensions(&storage, extensions);

    let src = fs::read_to_string(&args[1])?;
    if let Ok(script_blob) = Compiler::new(test_modules.iter().collect()).into_script_blob(&src) {
        let args: Vec<Vec<u8>> = vec![];
        let res = sess.execute_script(script_blob, vec![], args, &mut UnmeteredGasMeter)?;
        println!("{:?}", res);
    } else {
        let module = Compiler::new(test_modules.iter().collect()).into_compiled_module(&src)?;
        let mut module_blob = vec![];
        module.serialize(&mut module_blob)?;

        sess.publish_module(
            module_blob,
            *module.self_id().address(),
            &mut UnmeteredGasMeter,
        )?;
        let args: Vec<Vec<u8>> = vec![];
        let res = sess.execute_function_bypass_visibility(
            &module.self_id(),
            IdentStr::new("run").unwrap(),
            vec![],
            args,
            &mut UnmeteredGasMeter,
        )?;
        println!("{:?}", res);
    }

    Ok(())
}
