# Copyright (c) 2026-present K. S. Ernest (iFire) Lee & godot-onnx contributors
# SPDX-License-Identifier: Apache-2.0 OR MIT
@tool
extends EditorImportPlugin

func _get_importer_name() -> String:
	return "godot_onnx.onnx"

func _get_visible_name() -> String:
	return "ONNX Model"

func _get_recognized_extensions() -> PackedStringArray:
	return PackedStringArray(["onnx"])

func _get_resource_type() -> String:
	return "OnnxModelData"

func _get_save_extension() -> String:
	return "res"

func _import(source_file: String, save_path: String, options: Dictionary, platform_variants: Array, gen_files: Array) -> Error:
	var f := FileAccess.open(source_file, FileAccess.READ)
	if f == null:
		return Error.FAILED
	var bytes: PackedByteArray = f.get_buffer(f.get_length())
	f.close()
	if bytes.is_empty():
		return Error.FAILED
	var res := ClassDB.instantiate("OnnxModelData") as OnnxModelData
	if res == null:
		return Error.FAILED
	res.set_data(bytes)
	var out_path := save_path + "." + _get_save_extension()
	return ResourceSaver.save(res, out_path)
