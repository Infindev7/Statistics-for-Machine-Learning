# Copyright (c) 2026-present K. S. Ernest (iFire) Lee & godot-onnx contributors
# SPDX-License-Identifier: Apache-2.0 OR MIT
@tool
extends EditorPlugin

var _import_plugin: EditorImportPlugin

func _enter_tree() -> void:
	_import_plugin = preload("import_plugin.gd").new()
	add_import_plugin(_import_plugin)

func _exit_tree() -> void:
	remove_import_plugin(_import_plugin)
	_import_plugin = null
