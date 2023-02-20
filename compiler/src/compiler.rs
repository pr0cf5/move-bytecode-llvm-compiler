use std::collections::BTreeMap;

use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetTriple,
};
use inkwell::types::{BasicType, BasicTypeEnum};
use inkwell::values::{BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};
use move_binary_format::control_flow_graph::ControlFlowGraph;
use move_binary_format::file_format::{FunctionDefinition, SignatureToken};
use move_binary_format::{
    access::ModuleAccess,
    binary_views::FunctionView,
    file_format::{Bytecode, CompiledModule, FunctionDefinitionIndex},
};

pub struct TranslationUnit<'a> {
    context: &'a Context,
    module: Module<'a>,
    builder: Builder<'a>,
}

type Bid = u16;
type Iid = u16;
type Lid = usize;
type Gid = usize;

struct CfgInfo {
    block_map: BTreeMap<Bid, Iid>,
    entry_bid: Bid,
}

struct BuildMeta<'a> {
    function: FunctionValue<'a>,
    block_map: BTreeMap<Bid, (BasicBlock<'a>, bool)>,
    param_map: BTreeMap<Lid, PointerValue<'a>>,
    local_map: BTreeMap<Lid, PointerValue<'a>>,
    global_map: BTreeMap<Lid, PointerValue<'a>>,
    stack: Vec<BasicValueEnum<'a>>,
}

fn vector_type<'a>(context: &'a Context, inner_llvm_type: BasicTypeEnum<'a>) -> BasicTypeEnum<'a> {
    let length_type = context.i64_type();

    // Define a new structure type that contains a pointer to an array and a length field
    let vector_type = context.struct_type(
        &[
            inner_llvm_type
                .ptr_type(AddressSpace::default())
                .as_basic_type_enum(),
            length_type.as_basic_type_enum(),
        ],
        false,
    );
    vector_type
        .ptr_type(AddressSpace::default())
        .as_basic_type_enum()
}

impl<'a> BuildMeta<'a> {
    fn new(function: FunctionValue<'a>) -> Self {
        Self {
            function,
            block_map: BTreeMap::new(),
            param_map: BTreeMap::new(),
            local_map: BTreeMap::new(),
            global_map: BTreeMap::new(),
            stack: vec![],
        }
    }

    fn get_local_reference(&self, lid: Lid) -> Option<&PointerValue<'a>> {
        let params_len = self.function.count_params() as usize;
        if lid < params_len {
            self.param_map.get(&lid)
        } else {
            let fixed_lid = lid - params_len;
            self.local_map.get(&fixed_lid)
        }
    }

    fn get_global(&self, gid: Gid) -> Option<&PointerValue<'a>> {
        self.global_map.get(&gid)
    }
}

impl CfgInfo {
    fn new(
        compiled_module: &CompiledModule,
        func_def_index: FunctionDefinitionIndex,
    ) -> anyhow::Result<Self> {
        let mut block_map = BTreeMap::new();
        let func_def = compiled_module.function_def_at(func_def_index);
        let function_handle = compiled_module.function_handle_at(func_def.function);
        let view = FunctionView::function(
            compiled_module,
            func_def_index,
            func_def.code.as_ref().unwrap(),
            function_handle,
        );
        let cfg = view.cfg();
        for bid in cfg.blocks() {
            let start = cfg.block_start(bid);
            let end = cfg.block_end(bid);
            block_map.insert(start, end);
        }
        let entry_bid = cfg.entry_block_id();
        Ok(CfgInfo {
            block_map,
            entry_bid,
        })
    }
}

enum BlockExit<'a> {
    Uncond {
        target: Bid,
    },
    Cond {
        predicate: IntValue<'a>,
        target_if: Bid,
        target_else: Bid,
    },
    Ret,
}

impl<'a> TranslationUnit<'a> {
    pub fn new(context: &'a Context, name: &str) -> Self {
        Self {
            context,
            module: context.create_module(name),
            builder: context.create_builder(),
        }
    }

    fn init_helper_functions(&self) {
        let param_types = vec![self.context.i64_type().into()];
        let function_type = self.context.i64_type().fn_type(&param_types, false);
        self.module
            .add_function("vector_create", function_type, Some(Linkage::DLLImport));

        let param_types = vec![self.context.i64_type().into()];
        let function_type = self.context.i64_type().fn_type(&param_types, false);
        self.module
            .add_function("vector_length", function_type, Some(Linkage::DLLImport));

        let param_types = vec![
            self.context.i64_type().into(),
            self.context.i64_type().into(),
        ];
        let function_type = self.context.i64_type().fn_type(&param_types, false);
        self.module
            .add_function("vector_get", function_type, Some(Linkage::DLLImport));

        let param_types = vec![
            self.context.i64_type().into(),
            self.context.i64_type().into(),
        ];
        let function_type = self.context.i64_type().fn_type(&param_types, false);
        self.module
            .add_function("vector_push", function_type, Some(Linkage::DLLImport));

        let param_types = vec![self.context.i64_type().into()];
        let function_type = self.context.i64_type().fn_type(&param_types, false);
        self.module
            .add_function("vector_pop", function_type, Some(Linkage::DLLImport));

        let function_type = self.context.i64_type().fn_type(&[], false);
        self.module
            .add_function("abort", function_type, Some(Linkage::DLLImport));
    }

    fn build_store_to_local(
        &self,
        true_lid: Lid,
        value: BasicValueEnum<'a>,
        build_meta: &mut BuildMeta<'a>,
    ) -> anyhow::Result<()> {
        let &ptr = build_meta.local_map.get(&true_lid).unwrap();
        self.builder.build_store(ptr, value);
        Ok(())
    }

    fn build_load_from_local(
        &self,
        lid: Lid,
        build_meta: &mut BuildMeta<'a>,
    ) -> anyhow::Result<BasicValueEnum<'a>> {
        let &ptr = build_meta.get_local_reference(lid).unwrap();
        Ok(self.builder.build_load(ptr, ""))
    }

    fn translate_instruction(
        &self,
        move_module: &CompiledModule,
        bytecode: &[Bytecode],
        iid: Iid,
        build_meta: &mut BuildMeta<'a>,
    ) -> anyhow::Result<()> {
        let insn = &bytecode[iid as usize];
        match insn {
            Bytecode::ImmBorrowLoc(lid) | Bytecode::MutBorrowLoc(lid) => {
                let ptr = build_meta.get_local_reference(*lid as Lid).unwrap();
                build_meta.stack.push(ptr.as_basic_value_enum());
            }
            Bytecode::MoveLoc(lid) => {
                let ptr = build_meta.get_local_reference(*lid as Lid).unwrap();
                let value = self.builder.build_load(*ptr, "");
                build_meta.stack.push(value);
            }
            Bytecode::StLoc(lid) => {
                let top_value = build_meta.stack.pop().unwrap();
                let true_lid = *lid as usize - build_meta.function.count_params() as usize;
                self.build_store_to_local(true_lid, top_value, build_meta)?;
            }
            Bytecode::CopyLoc(lid) => {
                let value = self.build_load_from_local(*lid as Lid, build_meta)?;
                build_meta.stack.push(value);
            }
            Bytecode::VecLen(_) => {
                // it's a reference to a vector pointer, so one deref is required
                let vec_ptr_ptr = build_meta.stack.pop().unwrap().into_pointer_value();
                let vec_ptr = self
                    .builder
                    .build_load(vec_ptr_ptr, "")
                    .into_pointer_value();
                let vec_ptr_in_int =
                    self.builder
                        .build_ptr_to_int(vec_ptr, self.context.i64_type(), "");
                let return_value = self.builder.build_call(
                    self.module.get_function("vector_length").unwrap(),
                    &[vec_ptr_in_int.into()],
                    "",
                );
                let length = return_value.try_as_basic_value().unwrap_left();
                build_meta.stack.push(length);
            }
            Bytecode::LdU8(c) => build_meta.stack.push(
                self.context
                    .i8_type()
                    .const_int(*c as u64, false)
                    .as_basic_value_enum(),
            ),
            Bytecode::LdU64(c) => build_meta.stack.push(
                self.context
                    .i64_type()
                    .const_int(*c, false)
                    .as_basic_value_enum(),
            ),
            Bytecode::LdU128(c) => {
                let bits = [*c as u64, (c >> 64) as u64];
                build_meta.stack.push(
                    self.context
                        .i128_type()
                        .const_int_arbitrary_precision(&bits)
                        .as_basic_value_enum(),
                )
            }
            Bytecode::Neq => {
                let top = build_meta.stack.pop().unwrap();
                let second_top = build_meta.stack.pop().unwrap();
                assert!(top.is_int_value());
                assert!(second_top.is_int_value());
                let predicate = self.builder.build_int_compare(
                    IntPredicate::NE,
                    second_top.into_int_value(),
                    top.into_int_value(),
                    "",
                );
                build_meta.stack.push(predicate.as_basic_value_enum());
            }
            Bytecode::Eq => {
                let top = build_meta.stack.pop().unwrap();
                let second_top = build_meta.stack.pop().unwrap();
                assert!(top.is_int_value());
                assert!(second_top.is_int_value());
                let predicate = self.builder.build_int_compare(
                    IntPredicate::EQ,
                    second_top.into_int_value(),
                    top.into_int_value(),
                    "",
                );
                build_meta.stack.push(predicate.as_basic_value_enum());
            }
            Bytecode::Lt => {
                let top = build_meta.stack.pop().unwrap();
                let second_top = build_meta.stack.pop().unwrap();
                assert!(top.is_int_value());
                assert!(second_top.is_int_value());
                let predicate = self.builder.build_int_compare(
                    IntPredicate::NE,
                    second_top.into_int_value(),
                    top.into_int_value(),
                    "",
                );
                build_meta.stack.push(predicate.as_basic_value_enum());
            }
            Bytecode::VecImmBorrow(_) => {
                let top = build_meta.stack.pop().unwrap(); // vector index
                let second_top = build_meta.stack.pop().unwrap(); // vector (struct vector **)
                assert!(top.is_int_value());
                assert!(second_top.is_pointer_value());
                let struct_ptr = self
                    .builder
                    .build_load(second_top.into_pointer_value(), "")
                    .into_pointer_value();
                let data_field_ptr_ptr = self.builder.build_struct_gep(struct_ptr, 0, "").unwrap();
                let data_field_ptr = self
                    .builder
                    .build_load(data_field_ptr_ptr, "")
                    .into_pointer_value();
                let elem_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(data_field_ptr, &[top.into_int_value()], "")
                };
                // T *
                build_meta.stack.push(elem_ptr.as_basic_value_enum());
            }
            Bytecode::ReadRef => {
                let ptr = build_meta.stack.pop().unwrap();
                assert!(ptr.is_pointer_value());
                let value = self.builder.build_load(ptr.into_pointer_value(), "");
                build_meta.stack.push(value);
            }
            Bytecode::CastU8 => {
                let top = build_meta.stack.pop().unwrap();
                assert!(top.is_int_value());
                let value =
                    self.builder
                        .build_int_cast(top.into_int_value(), self.context.i8_type(), "");
                build_meta.stack.push(value.as_basic_value_enum());
            }
            Bytecode::CastU64 => {
                let top = build_meta.stack.pop().unwrap();
                assert!(top.is_int_value());
                let value =
                    self.builder
                        .build_int_cast(top.into_int_value(), self.context.i64_type(), "");
                build_meta.stack.push(value.as_basic_value_enum());
            }
            Bytecode::CastU128 => {
                let top = build_meta.stack.pop().unwrap();
                assert!(top.is_int_value());
                let value =
                    self.builder
                        .build_int_cast(top.into_int_value(), self.context.i128_type(), "");
                build_meta.stack.push(value.as_basic_value_enum());
            }
            Bytecode::Shl => {
                let top = build_meta.stack.pop().unwrap(); // shift amount
                let second_top = build_meta.stack.pop().unwrap(); // value shifted
                assert!(top.is_int_value());
                assert!(second_top.is_int_value());
                let shift_amount = self.builder.build_int_cast(
                    top.into_int_value(),
                    second_top.get_type().into_int_type(),
                    "",
                );
                let value =
                    self.builder
                        .build_left_shift(second_top.into_int_value(), shift_amount, "");
                build_meta.stack.push(value.as_basic_value_enum());
            }
            Bytecode::Shr => {
                let top = build_meta.stack.pop().unwrap(); // shift amount
                let second_top = build_meta.stack.pop().unwrap(); // value shifted
                assert!(top.is_int_value());
                assert!(second_top.is_int_value());
                let shift_amount = self.builder.build_int_cast(
                    top.into_int_value(),
                    second_top.get_type().into_int_type(),
                    "",
                );
                let value = self.builder.build_right_shift(
                    second_top.into_int_value(),
                    shift_amount,
                    false,
                    "",
                );
                build_meta.stack.push(value.as_basic_value_enum());
            }
            Bytecode::BitOr => {
                let top = build_meta.stack.pop().unwrap();
                let second_top = build_meta.stack.pop().unwrap();
                let value =
                    self.builder
                        .build_or(second_top.into_int_value(), top.into_int_value(), "");
                build_meta.stack.push(value.as_basic_value_enum());
            }
            Bytecode::BitAnd => {
                let top = build_meta.stack.pop().unwrap();
                let second_top = build_meta.stack.pop().unwrap();
                assert!(top.is_int_value() && second_top.is_int_value());
                let value =
                    self.builder
                        .build_and(second_top.into_int_value(), top.into_int_value(), "");
                build_meta.stack.push(value.as_basic_value_enum());
            }
            Bytecode::Add => {
                let top = build_meta.stack.pop().unwrap();
                let second_top = build_meta.stack.pop().unwrap();
                assert!(top.is_int_value() && second_top.is_int_value());
                if top.into_int_value() == self.context.i8_type().const_int(1, false)
                    && second_top.into_int_value() == self.context.i8_type().const_int(255, false)
                {
                    // abort sequence
                    self.builder
                        .build_call(self.module.get_function("abort").unwrap(), &[], "");
                }
                let value = self.builder.build_int_add(
                    second_top.into_int_value(),
                    top.into_int_value(),
                    "",
                );
                build_meta.stack.push(value.as_basic_value_enum());
            }
            Bytecode::Sub => {
                let top = build_meta.stack.pop().unwrap();
                let second_top = build_meta.stack.pop().unwrap();
                let value = self.builder.build_int_sub(
                    second_top.into_int_value(),
                    top.into_int_value(),
                    "",
                );
                build_meta.stack.push(value.as_basic_value_enum());
            }
            Bytecode::Xor => {
                let top = build_meta.stack.pop().unwrap();
                let second_top = build_meta.stack.pop().unwrap();
                let value =
                    self.builder
                        .build_xor(second_top.into_int_value(), top.into_int_value(), "");
                build_meta.stack.push(value.as_basic_value_enum());
            }
            Bytecode::LdConst(c) => {
                let const_pool_entry = move_module.constant_at(*c);
                let llvm_type = self.translate_move_type(move_module, &const_pool_entry.type_)?;
                let gid = self.get_or_add_global(c.0 as usize, llvm_type, build_meta)?;
                let global_ptr = build_meta.get_global(gid).unwrap();
                let loaded_constant = self.builder.build_load(*global_ptr, "");
                build_meta.stack.push(loaded_constant);
            }
            Bytecode::VecPack(sig_idx, length) => {
                let sig = &move_module.signature_at(*sig_idx).0[0];
                assert!(*length == 0);
                let inner_llvm_type = self.translate_move_type(move_module, sig)?;
                let vector_type_ptr = vector_type(&self.context, inner_llvm_type);
                let return_value = self.builder.build_call(
                    self.module.get_function("vector_create").unwrap(),
                    &[self.context.i64_type().const_int(*length, false).into()],
                    "",
                );
                let value = self.builder.build_int_to_ptr(
                    return_value
                        .try_as_basic_value()
                        .unwrap_left()
                        .into_int_value(),
                    vector_type_ptr.into_pointer_type(),
                    "",
                );
                build_meta.stack.push(value.as_basic_value_enum());
            }
            Bytecode::VecPushBack(_) => {
                let top = build_meta.stack.pop().unwrap(); // element to push
                let second_top = build_meta.stack.pop().unwrap(); // vector reference
                let vector_ptr = self.builder.build_load(second_top.into_pointer_value(), "");
                let vector_ptr_in_int = self.builder.build_ptr_to_int(
                    vector_ptr.into_pointer_value(),
                    self.context.i64_type(),
                    "",
                );
                self.builder.build_call(
                    self.module.get_function("vector_push").unwrap(),
                    &[vector_ptr_in_int.into(), top.into()],
                    "",
                );
            }
            Bytecode::VecPopBack(_) => {
                let top = build_meta.stack.pop().unwrap(); // reference to vector
                let vector_ptr = self.builder.build_load(top.into_pointer_value(), "");
                let vector_ptr_in_int = self.builder.build_ptr_to_int(
                    vector_ptr.into_pointer_value(),
                    self.context.i64_type(),
                    "",
                );
                let return_value = self.builder.build_call(
                    self.module.get_function("vector_pop").unwrap(),
                    &[vector_ptr_in_int.into()],
                    "",
                );
                let popped_value = return_value.try_as_basic_value().unwrap_left();
                build_meta.stack.push(popped_value);
            }
            Bytecode::Pop => {
                build_meta.stack.pop().unwrap();
            }
            _ => panic!("unimplemented insn: {:?}", insn),
        }
        Ok(())
    }

    fn translate_block_exit(
        &self,
        move_module: &CompiledModule,
        bytecode: &[Bytecode],
        iid: Iid,
        build_meta: &mut BuildMeta<'a>,
    ) -> anyhow::Result<BlockExit<'a>> {
        let insn = &bytecode[iid as usize];
        match insn {
            Bytecode::BrTrue(dest) => {
                let predicate = build_meta.stack.pop().unwrap();
                assert!(predicate.is_int_value());
                Ok(BlockExit::Cond {
                    predicate: predicate.into_int_value(),
                    target_if: *dest,
                    target_else: iid + 1,
                })
            }
            Bytecode::BrFalse(dest) => {
                let predicate = build_meta.stack.pop().unwrap();
                assert!(predicate.is_int_value());
                let inverted_predicate = self.builder.build_not(predicate.into_int_value(), "");
                Ok(BlockExit::Cond {
                    predicate: inverted_predicate,
                    target_if: *dest,
                    target_else: iid + 1,
                })
            }
            Bytecode::Branch(dest) => Ok(BlockExit::Uncond { target: *dest }),
            Bytecode::Ret => Ok(BlockExit::Ret),
            _ => {
                // sometimes, a block exit is a regular instruction
                self.translate_instruction(move_module, bytecode, iid, build_meta)?;
                Ok(BlockExit::Uncond { target: iid + 1 })
            }
        }
    }

    fn translate_block(
        &self,
        move_module: &CompiledModule,
        bytecode: &[Bytecode],
        start_iid: Iid,
        cfg_info: &CfgInfo,
        build_meta: &mut BuildMeta<'a>,
    ) -> anyhow::Result<Option<BlockExit<'a>>> {
        let (bb, compiled) = build_meta.block_map.get(&start_iid).unwrap();
        if *compiled {
            return Ok(None);
        }
        let &end_iid = cfg_info.block_map.get(&start_iid).unwrap();

        self.builder.position_at_end(*bb);
        for iid in start_iid..end_iid {
            self.translate_instruction(move_module, bytecode, iid, build_meta)?;
        }

        let exit = self.translate_block_exit(move_module, bytecode, end_iid, build_meta)?;
        Ok(Some(exit))
    }

    fn translate_move_type(
        &self,
        move_module: &CompiledModule,
        sig: &SignatureToken,
    ) -> anyhow::Result<BasicTypeEnum<'a>> {
        match sig {
            SignatureToken::Bool => Ok(self.context.bool_type().as_basic_type_enum()),
            SignatureToken::U8 => Ok(self.context.i8_type().as_basic_type_enum()),
            SignatureToken::U16 => Ok(self.context.i16_type().as_basic_type_enum()),
            SignatureToken::U32 => Ok(self.context.i32_type().as_basic_type_enum()),
            SignatureToken::U64 => Ok(self.context.i64_type().as_basic_type_enum()),
            SignatureToken::U128 => Ok(self.context.i128_type().as_basic_type_enum()),
            SignatureToken::U256 => todo!(),
            SignatureToken::MutableReference(inner_sig) | SignatureToken::Reference(inner_sig) => {
                let inner_llvm_type = self.translate_move_type(move_module, inner_sig)?;
                Ok(inner_llvm_type
                    .ptr_type(AddressSpace::default())
                    .as_basic_type_enum())
            }
            SignatureToken::Vector(inner_sig) => {
                let inner_llvm_type = self.translate_move_type(move_module, inner_sig)?;
                Ok(vector_type(&self.context, inner_llvm_type))
            }
            _ => panic!("unimplemented: {:?}", sig),
        }
    }

    fn add_local(
        &self,
        true_lid: Lid,
        llvm_type: BasicTypeEnum<'a>,
        build_meta: &mut BuildMeta<'a>,
    ) -> anyhow::Result<Lid> {
        let ptr = self
            .builder
            .build_alloca::<BasicTypeEnum>(llvm_type, &format!("loc{}", true_lid));
        build_meta.local_map.insert(true_lid, ptr);
        Ok(true_lid)
    }

    fn add_param(
        &self,
        param_lid: Lid,
        llvm_type: BasicTypeEnum<'a>,
        build_meta: &mut BuildMeta<'a>,
    ) -> anyhow::Result<Lid> {
        let ptr = self
            .builder
            .build_alloca::<BasicTypeEnum>(llvm_type, &format!("param{}", param_lid));
        build_meta.param_map.insert(param_lid, ptr);
        Ok(param_lid)
    }

    fn add_global(
        &self,
        gid: Gid,
        llvm_type: BasicTypeEnum<'a>,
        build_meta: &mut BuildMeta<'a>,
    ) -> anyhow::Result<Gid> {
        let ptr = self
            .module
            .add_global(llvm_type, None, &format!("global{}", gid));
        build_meta.global_map.insert(gid, ptr.as_pointer_value());
        Ok(gid)
    }

    fn get_or_add_global(
        &self,
        gid: Gid,
        llvm_type: BasicTypeEnum<'a>,
        build_meta: &mut BuildMeta<'a>,
    ) -> anyhow::Result<Gid> {
        if let Some(_) = build_meta.global_map.get(&gid) {
            Ok(gid)
        } else {
            self.add_global(gid, llvm_type, build_meta)
        }
    }

    fn translate_intro(
        &self,
        move_module: &CompiledModule,
        function_def: &FunctionDefinition,
        build_meta: &mut BuildMeta<'a>,
    ) -> anyhow::Result<()> {
        // initialize parameters
        for (i, param) in build_meta.function.get_params().iter().enumerate() {
            self.add_param(i, param.get_type(), build_meta)?;
            let param_ptr = build_meta.param_map.get(&i).unwrap();
            self.builder.build_store(*param_ptr, *param);
        }
        // initialize locals
        for (i, sig) in move_module
            .signature_at(function_def.code.as_ref().unwrap().locals)
            .0
            .iter()
            .enumerate()
        {
            let llvm_type = self.translate_move_type(move_module, &sig)?;
            self.add_local(i, llvm_type, build_meta)?;
        }
        Ok(())
    }

    fn translate_function(
        &self,
        move_module: &CompiledModule,
        function_index: FunctionDefinitionIndex,
    ) -> anyhow::Result<()> {
        let function_def = move_module.function_def_at(function_index);
        let function_handle = move_module.function_handle_at(function_def.function);
        let function_name = move_module.identifier_at(function_handle.name);
        let function_params = &move_module.signature_at(function_handle.parameters).0;
        let function_return_type = &move_module.signature_at(function_handle.return_).0;
        assert_eq!(function_return_type.len(), 0);
        let function_code = function_def.code.as_ref().unwrap();
        let cfg_info = CfgInfo::new(&move_module, function_index)?;

        let mut llvm_function_param_types = Vec::new();
        for sig in function_params {
            let llvm_type = self.translate_move_type(move_module, sig)?;
            llvm_function_param_types.push(llvm_type.into());
        }
        let llvm_function_return_type = self.context.void_type();
        let llvm_function_type =
            llvm_function_return_type.fn_type(&llvm_function_param_types, false);
        let llvm_function = self.module.add_function(
            function_name.as_str(),
            llvm_function_type,
            Some(Linkage::External),
        );

        /* create initial block */

        let mut build_meta = BuildMeta::new(llvm_function);

        let intro_bb = self.context.append_basic_block(llvm_function, "intro");
        self.builder.position_at_end(intro_bb);
        self.translate_intro(move_module, function_def, &mut build_meta)?;

        let start_iid = cfg_info.entry_bid;
        let mut stack = vec![start_iid];

        /* first, create all the blocks */
        for (bid, _) in cfg_info.block_map.iter() {
            let bb = self
                .context
                .append_basic_block(build_meta.function, &format!("block_{}", start_iid));
            build_meta.block_map.insert(*bid, (bb, false));
        }
        let entry_bb = build_meta.block_map.get(&cfg_info.entry_bid).unwrap().0;

        while let Some(bid) = stack.pop() {
            if let Some(exit) = self.translate_block(
                move_module,
                &function_code.code,
                bid,
                &cfg_info,
                &mut build_meta,
            )? {
                build_meta.block_map.get_mut(&bid).unwrap().1 = true;
                match exit {
                    BlockExit::Uncond { target } => {
                        let (block, complete) = build_meta.block_map.get(&target).unwrap();
                        self.builder.build_unconditional_branch(*block);
                        if !complete {
                            stack.push(target);
                        }
                    }
                    BlockExit::Cond {
                        predicate,
                        target_if,
                        target_else,
                    } => {
                        let (block_if, if_complete) = build_meta.block_map.get(&target_if).unwrap();
                        let (block_else, else_complete) =
                            build_meta.block_map.get(&target_else).unwrap();
                        self.builder
                            .build_conditional_branch(predicate, *block_if, *block_else);
                        if !if_complete {
                            stack.push(target_if);
                        }
                        if !else_complete {
                            stack.push(target_else);
                        }
                    }
                    BlockExit::Ret => {
                        self.builder.build_return(None);
                    }
                }
            }
        }

        self.builder.position_at_end(intro_bb);
        self.builder.build_unconditional_branch(entry_bb);
        Ok(())
    }

    pub fn translate(&self, move_module: &CompiledModule) -> anyhow::Result<Vec<u8>> {
        self.init_helper_functions();
        for index in 0..move_module.function_defs().len() {
            self.translate_function(
                &move_module,
                FunctionDefinitionIndex(index.try_into().unwrap()),
            )?;
        }
        Target::initialize_x86(&InitializationConfig::default());
        let target = Target::from_name("x86-64").expect("x86-64 unsupported");
        let target_machine = target
            .create_target_machine(
                &TargetTriple::create("x86_64-pc-linux-gnu"),
                "x86-64",
                "",
                OptimizationLevel::Aggressive,
                RelocMode::Static,
                CodeModel::Large,
            )
            .expect("Failed to initialize target machine");
        let object_code = target_machine
            .write_to_memory_buffer(&self.module, FileType::Object)
            .expect("Failed to generate machine code")
            .as_slice()
            .to_vec();
        Ok(object_code)
    }
}
