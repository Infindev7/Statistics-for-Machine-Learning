extends Node3D
## Terramorph ,  live VAE terrain morphing + walkable bean explorer.
## Attach to a Node3D in an otherwise empty 3D scene and press F5.
##
## Requires:
##   - godot-onnx addon installed (addons/godot-onnx/ + godot-onnx.gdextension at project root)
##   - res://models/terrain_decoder.onnx (single-file, weights merged in)
##   - res://terrain.gdshader (next to this script)
##
## Controls: drag sliders to morph. Left/right-drag = orbit, wheel = zoom.
## "Walk mode" spawns a bean: WASD/arrows to move, Space to jump.

const LATENT_DIM := 5
const TEX_SIZE := 128          # must match the decoder's output (1, 1, TEX_SIZE, TEX_SIZE)
const PLANE_SIZE := 240.0
const GRID_SUBDIV := 160
const MODEL_PATH := "res://models/terrain_decoder.onnx"

var _panel: PanelContainer
var _module: OnnxModule
var _latent := PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0])
var _sliders: Array[HSlider] = []
var _val_labels: Array[Label] = []
var _height_tex: ImageTexture
var _height_img: Image
var _mat: ShaderMaterial
var _dirty := true
var _time_val: Label

# Shaping params ,  single source of truth, pushed to the shader on ready
# and used for CPU ground-height queries (bean physics).
var _sharpness := 1.6
var _ridge := 0.35
var _groove := 0.45
var _ridge_scale_f := 44.0
var _warp := 1.2
var _hscale := 22.0
var _crest_thresh := 0.30
var _groove_thresh := 0.34
var _walk_btn: Button

var _sun: DirectionalLight3D
var _sky_mat: ProceduralSkyMaterial
var _env: Environment
var _cam: Camera3D

var _cam_yaw := 0.7
var _cam_pitch := 0.42
var _cam_dist := 95.0
var _cam_target := Vector3(0, 5, 0)
var _orbiting := false

var _walk_mode := false
var _bean: Node3D
var _bean_body: MeshInstance3D
var _bean_pos := Vector3.ZERO
var _bean_vel := Vector3.ZERO
var _bean_vy := 0.0
var _bean_yaw := 0.0
var _view_yaw := 0.0
var _view_pitch := 0.0
var _follow_dist := 10.0
var _grounded := true


func _ready() -> void:
	_module = OnnxModule.new()
	_module.load(MODEL_PATH)
	if not _module.is_loaded():
		push_error("Could not load " + MODEL_PATH + " ,  is the single-file .onnx in res://models/?")
		return
	_build_world()
	_build_ui()
	_on_time_changed(10.0)


func _build_world() -> void:
	_sun = DirectionalLight3D.new()
	_sun.shadow_enabled = true
	_sun.directional_shadow_max_distance = 400.0
	add_child(_sun)

	_env = Environment.new()
	_env.background_mode = Environment.BG_SKY
	_env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	_env.ambient_light_sky_contribution = 0.6
	# Subtle distance haze for scale: depth-gated so nearby terrain stays crisp.
	#_env.fog_enabled = true
	#_env.fog_mode = Environment.FOG_MODE_DEPTH
	#_env.fog_depth_begin = 160.0
	#_env.fog_depth_end = 600.0
	#_env.fog_sky_affect = 0.0

	var sky := Sky.new()
	_sky_mat = ProceduralSkyMaterial.new()
	sky.sky_material = _sky_mat
	_env.sky = sky
	var world_env := WorldEnvironment.new()
	world_env.environment = _env
	add_child(world_env)

	_cam = Camera3D.new()
	add_child(_cam)
	_cam.current = true
	_update_camera()

	# Self-built grid: UVs are exactly (x/size + 0.5, z/size + 0.5), so the
	# GDScript ground-height query matches the shader displacement 1:1.
	var mi := MeshInstance3D.new()
	mi.mesh = _build_grid_mesh(GRID_SUBDIV)
	_mat = ShaderMaterial.new()
	_mat.shader = preload("res://terrain.gdshader")
	mi.material_override = _mat
	add_child(mi)

	var img := Image.create(TEX_SIZE, TEX_SIZE, false, Image.FORMAT_RF)
	img.fill(Color(0, 0, 0, 1))
	_height_img = img
	_height_tex = ImageTexture.create_from_image(img)
	_mat.set_shader_parameter("heightmap", _height_tex)
	_mat.set_shader_parameter("plane_size", PLANE_SIZE)
	_mat.set_shader_parameter("height_scale", _hscale)
	_mat.set_shader_parameter("sharpness", _sharpness)
	_mat.set_shader_parameter("ridge_amt", _ridge)
	_mat.set_shader_parameter("groove_amt", _groove)
	_mat.set_shader_parameter("ridge_scale", _ridge_scale_f)
	_mat.set_shader_parameter("warp_amt", _warp)
	_mat.set_shader_parameter("crest_thresh", _crest_thresh)
	_mat.set_shader_parameter("groove_thresh", _groove_thresh)


func _build_grid_mesh(subdiv: int) -> ArrayMesh:
	var verts := PackedVector3Array()
	var uvs := PackedVector2Array()
	var indices := PackedInt32Array()
	for iz in range(subdiv + 1):
		for ix in range(subdiv + 1):
			var x := (float(ix) / subdiv - 0.5) * PLANE_SIZE
			var z := (float(iz) / subdiv - 0.5) * PLANE_SIZE
			verts.append(Vector3(x, 0.0, z))
			uvs.append(Vector2(float(ix) / subdiv, float(iz) / subdiv))
	for iz in range(subdiv):
		for ix in range(subdiv):
			var a := iz * (subdiv + 1) + ix
			var b := a + 1
			var c := a + subdiv + 1
			var d := c + 1
			indices.append_array([a, b, c, b, d, c])
	var arr := []
	arr.resize(Mesh.ARRAY_MAX)
	arr[Mesh.ARRAY_VERTEX] = verts
	arr[Mesh.ARRAY_TEX_UV] = uvs
	arr[Mesh.ARRAY_INDEX] = indices
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arr)
	return mesh


# ---------------------------------------------------------------- UI ---

func _build_ui() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)

	var panel := PanelContainer.new()
	panel.anchor_left = 1.0
	panel.anchor_right = 1.0
	panel.anchor_top = 0.5
	panel.anchor_bottom = 0.5
	panel.offset_left = -500
	panel.offset_right = -200
	panel.offset_top = -300
	panel.offset_bottom = 300
	layer.add_child(panel)
	_panel = panel

	var margin := MarginContainer.new()
	for side in ["margin_left", "margin_right", "margin_top", "margin_bottom"]:
		margin.add_theme_constant_override(side, 12)
	panel.add_child(margin)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 8)
	margin.add_child(vbox)

	vbox.add_child(_section_title("Latent Space", 18))
	for i in LATENT_DIM:
		vbox.add_child(_latent_slider_row(i))

	var btn := Button.new()
	btn.focus_mode = Control.FOCUS_NONE
	btn.text = "Randomize"
	btn.pressed.connect(_on_randomize)
	vbox.add_child(btn)

	var walk_btn := Button.new()
	walk_btn.focus_mode = Control.FOCUS_NONE
	walk_btn.text = "Walk mode: OFF"
	walk_btn.pressed.connect(_on_walk_toggle.bind(walk_btn))
	_walk_btn = walk_btn
	vbox.add_child(walk_btn)

	vbox.add_child(HSeparator.new())
	vbox.add_child(_section_title("Mountain shape", 14))
	vbox.add_child(_labeled_slider("sharp", 1.0, 3.0, 0.05, 1.6, _on_sharpness_changed))
	vbox.add_child(_labeled_slider("ridge", 0.0, 0.8, 0.05, 0.35, _on_ridge_changed))
	vbox.add_child(_labeled_slider("groove", 0.0, 0.9, 0.05, 0.45, _on_groove_changed))

	vbox.add_child(HSeparator.new())
	vbox.add_child(_section_title("Scene", 14))
	vbox.add_child(_labeled_slider("time", 0.0, 24.0, 0.25, 10.0, _on_time_changed, "%.1fh", "_time_val"))

	var hint := Label.new()
	hint.text = "Drag: orbit   Wheel: zoom   Walk: mouse look, WASD move, Space jump, ESC exit"
	hint.add_theme_font_size_override("font_size", 12)
	hint.modulate = Color(0.7, 0.7, 0.7)
	vbox.add_child(hint)


func _section_title(text: String, size: int) -> Label:
	var l := Label.new()
	l.text = text
	l.add_theme_font_size_override("font_size", size)
	return l


func _latent_slider_row(i: int) -> HBoxContainer:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	var name_label := Label.new()
	name_label.text = "z[%d]" % i
	name_label.custom_minimum_size = Vector2(38, 0)
	var slider := HSlider.new()
	slider.min_value = -3.0
	slider.max_value = 3.0
	slider.step = 0.01
	slider.value = 0.0
	slider.custom_minimum_size = Vector2(170, 0)
	slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	var val := Label.new()
	val.text = "0.00"
	val.custom_minimum_size = Vector2(46, 0)
	slider.value_changed.connect(_on_latent_changed.bind(i, val))
	_sliders.append(slider)
	_val_labels.append(val)
	row.add_child(name_label)
	row.add_child(slider)
	row.add_child(val)
	return row


func _labeled_slider(label_text: String, min_v: float, max_v: float, step: float,
		default_v: float, on_change: Callable, fmt: String = "%.2f", store_as: String = "") -> HBoxContainer:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	var name_label := Label.new()
	name_label.text = label_text
	name_label.custom_minimum_size = Vector2(38, 0)
	var slider := HSlider.new()
	slider.min_value = min_v
	slider.max_value = max_v
	slider.step = step
	slider.value = default_v
	slider.custom_minimum_size = Vector2(170, 0)
	slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	var val := Label.new()
	val.text = fmt % default_v
	val.custom_minimum_size = Vector2(46, 0)
	if store_as != "":
		set(store_as, val)
	slider.value_changed.connect(_on_labeled_changed.bind(val, on_change, fmt))
	row.add_child(name_label)
	row.add_child(slider)
	row.add_child(val)
	return row


func _on_latent_changed(value: float, i: int, val_label: Label) -> void:
	val_label.text = "%.2f" % value
	_latent[i] = value
	_dirty = true


func _on_labeled_changed(value: float, val_label: Label, on_change: Callable, fmt: String) -> void:
	val_label.text = fmt % value
	on_change.call(value)


func _on_randomize() -> void:
	for i in LATENT_DIM:
		var v := randf_range(-2.0, 2.0)
		_latent[i] = v
		_sliders[i].set_value_no_signal(v)
		_val_labels[i].text = "%.2f" % v
	_dirty = true


func _on_sharpness_changed(value: float) -> void:
	_sharpness = value
	if _mat != null:
		_mat.set_shader_parameter("sharpness", value)


func _on_ridge_changed(value: float) -> void:
	_ridge = value
	if _mat != null:
		_mat.set_shader_parameter("ridge_amt", value)


func _on_groove_changed(value: float) -> void:
	_groove = value
	if _mat != null:
		_mat.set_shader_parameter("groove_amt", value)


func _on_time_changed(hour: float) -> void:
	if _time_val != null:
		_time_val.text = "%.1fh" % hour
	var elev := sin(clampf((hour - 6.0) / 12.0, 0.0, 1.0) * PI)  # 0..1..0
	var day := clampf(elev * 4.0, 0.0, 1.0)
	_sun.rotation_degrees = Vector3(-lerpf(-8.0, 58.0, clampf(elev, 0.0, 1.0)),
		lerpf(-100.0, 100.0, hour / 24.0), 0.0)
	_sun.light_energy = lerpf(0.0, 1.2, day)
	_sun.light_color = Color(1.0, lerpf(0.5, 1.0, day), lerpf(0.35, 0.95, day))
	var night := 1.0 - day
	_sky_mat.sky_top_color = Color(lerpf(0.28, 0.01, night), lerpf(0.55, 0.02, night), lerpf(0.95, 0.06, night))
	_sky_mat.sky_horizon_color = Color(lerpf(0.75, 0.05, night), lerpf(0.72, 0.06, night), lerpf(0.85, 0.12, night))
	_sky_mat.ground_bottom_color = _sky_mat.sky_horizon_color.darkened(0.25)
	_sky_mat.ground_horizon_color = _sky_mat.sky_horizon_color
	_env.background_energy_multiplier = lerpf(0.12, 1.0, day)
	_env.ambient_light_energy = lerpf(0.3, 0.7, day)


# ---------------------------------------------------------------- bean ---

func _on_walk_toggle(btn: Button) -> void:
	_walk_mode = not _walk_mode
	btn.text = "Walk mode: ON" if _walk_mode else "Walk mode: OFF"
	_panel.visible = not _walk_mode
	if _walk_mode:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
		_spawn_bean()
		_view_yaw = _cam_yaw  # keep looking where you were looking
		_view_pitch = 0.0
	else:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		if is_instance_valid(_bean):
			_bean.queue_free()
			_bean = null
		_update_camera()


func _spawn_bean() -> void:
	_bean = Node3D.new()
	_bean_body = MeshInstance3D.new()
	var cap := CapsuleMesh.new()
	cap.radius = 0.35
	cap.height = 1.15
	_bean_body.mesh = cap
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(1.0, 0.55, 0.15)
	mat.roughness = 0.6
	_bean_body.material_override = mat
	_bean.add_child(_bean_body)
	# Eyes (bean faces +Z).
	for ex in [-0.14, 0.14]:
		var eye := MeshInstance3D.new()
		var sm := SphereMesh.new()
		sm.radius = 0.09
		sm.height = 0.18
		eye.mesh = sm
		var emat := StandardMaterial3D.new()
		emat.albedo_color = Color.WHITE
		eye.material_override = emat
		eye.position = Vector3(ex, 0.18, 0.30)
		_bean.add_child(eye)
		var pup := MeshInstance3D.new()
		var pm := SphereMesh.new()
		pm.radius = 0.04
		pm.height = 0.08
		pup.mesh = pm
		var pmat := StandardMaterial3D.new()
		pmat.albedo_color = Color(0.1, 0.1, 0.12)
		pup.material_override = pmat
		pup.position = Vector3(ex, 0.18, 0.375)
		_bean.add_child(pup)
	add_child(_bean)
	_bean.visible = false  # FPS: camera rides inside the head, no visible body
	_bean_pos = Vector3(0, 0, 8)
	_bean_pos.y = _terrain_height_at(_bean_pos) + 0.6
	_bean_yaw = _cam_yaw + PI  # face away from the camera
	_bean_vel = Vector3.ZERO
	_bean_vy = 0.0
	_bean.position = _bean_pos
	_bean.rotation.y = _bean_yaw


func _update_walk(delta: float) -> void:
	var strafe := 0.0
	var fwd_in := 0.0
	if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT):
		strafe += 1.0
	if Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT):
		strafe -= 1.0
	if Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP):
		fwd_in += 1.0
	if Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN):
		fwd_in -= 1.0
	var fwd := Vector3(-sin(_view_yaw), 0.0, -cos(_view_yaw))
	var right := Vector3(-fwd.z, 0.0, fwd.x)
	var wish := (fwd * fwd_in + right * strafe)
	if wish.length() > 1.0:
		wish = wish.normalized()
	_bean_vel = _bean_vel.lerp(wish * 9.0, 1.0 - exp(-10.0 * delta))
	_bean_pos += _bean_vel * delta
	_bean_pos.x = clampf(_bean_pos.x, -115.0, 115.0)
	_bean_pos.z = clampf(_bean_pos.z, -115.0, 115.0)

	# Gravity + jump, grounded on the terrain surface.
	var ground := _terrain_height_at(_bean_pos) + 0.6
	_bean_vy -= 24.0 * delta
	_bean_pos.y += _bean_vy * delta
	_grounded = false
	if _bean_pos.y <= ground:
		_bean_pos.y = ground
		_bean_vy = 0.0
		_grounded = true
	if _grounded and Input.is_key_pressed(KEY_SPACE):
		_bean_vy = 8.0
		_grounded = false

	# Face movement direction, lean into the run.
	var hspeed := Vector2(_bean_vel.x, _bean_vel.z).length()
	if hspeed > 0.6:
		_bean_yaw = lerp_angle(_bean_yaw, atan2(_bean_vel.x, _bean_vel.z), 1.0 - exp(-10.0 * delta))
	_bean.position = _bean_pos
	_bean.rotation.y = _bean_yaw
	_bean_body.rotation.x = lerpf(0.0, 0.18, clampf(hspeed / 9.0, 0.0, 1.0))

	# FPS camera at head height, looking along the view direction.
	# Lift over upcoming slopes so hillsides don't clip through the lens.
	var look_dir := Vector3(-sin(_view_yaw), 0.0, -cos(_view_yaw))
	var ahead_h := _terrain_height_at(_bean_pos + look_dir * 1.2)
	var head_y := maxf(_bean_pos.y + 1.05, ahead_h + 0.5)
	_cam.transform = Transform3D(
		Basis(Vector3.UP, _view_yaw) * Basis(Vector3.RIGHT, _view_pitch),
		Vector3(_bean_pos.x, head_y, _bean_pos.z))


# --------------------------------------- CPU mirror of terrain.gdshader ---

func _g_hash(p: Vector2) -> float:
	var x := sin(p.dot(Vector2(127.1, 311.7))) * 43758.5453123
	return x - floor(x)  # == GLSL fract()


func _g_vnoise(p: Vector2) -> float:
	var ix: float = floor(p.x)
	var iy: float = floor(p.y)
	var fx: float = p.x - ix
	var fy: float = p.y - iy
	var ux: float = fx * fx * (3.0 - 2.0 * fx)
	var uy: float = fy * fy * (3.0 - 2.0 * fy)
	var a := _g_hash(Vector2(ix, iy))
	var b := _g_hash(Vector2(ix + 1.0, iy))
	var c := _g_hash(Vector2(ix, iy + 1.0))
	var d := _g_hash(Vector2(ix + 1.0, iy + 1.0))
	return lerpf(lerpf(a, b, ux), lerpf(c, d, ux), uy)


func _g_ridged(p: Vector2) -> float:
	var q := Vector2(_g_vnoise(p), _g_vnoise(p + Vector2(5.2, 1.3)))
	var w := p + (q - Vector2(0.5, 0.5)) * 2.0 * _warp
	var sum := 0.0
	var amp := 0.55
	var pp := w
	for i in 4:
		var n := _g_vnoise(pp)
		n = 1.0 - absf(2.0 * n - 1.0)
		n = n * n
		sum += n * amp
		pp = pp * 2.03 + Vector2(1.7, 9.2)
		amp *= 0.5
	return clampf(sum, 0.0, 1.0)


func _sample_bilinear(uv: Vector2) -> float:
	var fx := uv.x * float(TEX_SIZE - 1)
	var fy := uv.y * float(TEX_SIZE - 1)
	var x0 := int(floor(fx))
	var y0 := int(floor(fy))
	var x1 := mini(x0 + 1, TEX_SIZE - 1)
	var y1 := mini(y0 + 1, TEX_SIZE - 1)
	var tx := fx - float(x0)
	var ty := fy - float(y0)
	var a := _height_img.get_pixel(x0, y0).r
	var b := _height_img.get_pixel(x1, y0).r
	var c := _height_img.get_pixel(x0, y1).r
	var d := _height_img.get_pixel(x1, y1).r
	return lerpf(lerpf(a, b, tx), lerpf(c, d, tx), ty)


func _terrain_height_at(p: Vector3) -> float:
	if _height_img == null:
		return 0.0
	var uv := Vector2(clampf(p.x / PLANE_SIZE + 0.5, 0.0, 1.0),
		clampf(p.z / PLANE_SIZE + 0.5, 0.0, 1.0))
	var h := _sample_bilinear(uv)
	var base: float = pow(clampf(h, 0.0, 1.0), _sharpness)
	var r := _g_ridged(uv * _ridge_scale_f)
	var thin := lerpf(1.0, 2.6, base)
	var crest := maxf(pow(r, thin) - _crest_thresh, 0.0) * _ridge
	var groove := maxf(_groove_thresh - r, 0.0) * _groove
	var mask: float = smoothstep(0.02, 0.3, base)
	return (base + (crest - groove) * mask) * _hscale


# ---------------------------------------------------------------- misc ---

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey:
		var ke := event as InputEventKey
		if ke.pressed and not ke.echo and ke.keycode == KEY_ESCAPE and _walk_mode:
			_on_walk_toggle(_walk_btn)
			get_viewport().set_input_as_handled()
		return
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_LEFT or mb.button_index == MOUSE_BUTTON_RIGHT:
			_orbiting = mb.pressed
		elif mb.button_index == MOUSE_BUTTON_WHEEL_UP and mb.pressed:
			if _walk_mode:
				_follow_dist = clampf(_follow_dist - 1.0, 4.0, 20.0)
			else:
				_cam_dist = clampf(_cam_dist - 2.5, 20.0, 400.0)
		elif mb.button_index == MOUSE_BUTTON_WHEEL_DOWN and mb.pressed:
			if _walk_mode:
				_follow_dist = clampf(_follow_dist + 1.0, 4.0, 20.0)
			else:
				_cam_dist = clampf(_cam_dist + 2.5, 20.0, 400.0)
	elif event is InputEventMouseMotion:
		var mm := event as InputEventMouseMotion
		if _walk_mode:
			# Captured mouse: look around freely, no dragging needed.
			_view_yaw -= mm.relative.x * 0.0032
			_view_pitch = clampf(_view_pitch - mm.relative.y * 0.0032, -1.35, 1.35)
		elif _orbiting:
			_cam_yaw -= mm.relative.x * 0.005
			_cam_pitch = clampf(_cam_pitch + mm.relative.y * 0.005, 0.06, 1.4)


func _update_camera() -> void:
	var off := Vector3(cos(_cam_pitch) * sin(_cam_yaw), sin(_cam_pitch),
		cos(_cam_pitch) * cos(_cam_yaw)) * _cam_dist
	_cam.look_at_from_position(_cam_target + off, _cam_target, Vector3.UP)


func _process(delta: float) -> void:
	if _walk_mode and is_instance_valid(_bean):
		_update_walk(delta)
	elif not _walk_mode:
		_update_camera()
	# Run inference at most once per frame, even while dragging.
	if _dirty and _module != null and _module.is_loaded():
		_dirty = false
		_run_inference()


func _run_inference() -> void:
	var dims := PackedInt64Array([1, LATENT_DIM])
	var input := OnnxTensor.from_float32s(_latent, dims)
	var result: Array = _module.call_module("", [input])
	if result.is_empty():
		return
	var out: OnnxTensor = result[0]
	var data := out.get_data().to_float32_array()
	if data.size() != TEX_SIZE * TEX_SIZE:
		push_warning("Unexpected decoder output size: %d" % data.size())
		return
	# Output layout is (1, 1, H, W) row-major -> pixel (x, y).
	var img := Image.create(TEX_SIZE, TEX_SIZE, false, Image.FORMAT_RF)
	var k := 0
	var min_h := INF
	var max_h := -INF
	for y in TEX_SIZE:
		for x in TEX_SIZE:
			var v := data[k]
			img.set_pixel(x, y, Color(v, 0, 0, 1))
			if v < min_h:
				min_h = v
			if v > max_h:
				max_h = v
			k += 1
	# Stretch to the full 0..1 range so every latent position gets the whole
	# vivid palette (deep water -> snow) instead of clustering in one dark band.
	# The 0.15 floor keeps near-flat outputs from amplifying noise into mountains.
	var span := maxf(max_h - min_h, 0.15)
	var inv := 1.0 / span
	for y in TEX_SIZE:
		for x in TEX_SIZE:
			var v := clampf((img.get_pixel(x, y).r - min_h) * inv, 0.0, 1.0)
			img.set_pixel(x, y, Color(v, 0, 0, 1))
	_height_img = img
	_height_tex.update(img)
