shader_type spatial;
render_mode cull_back;

// 128x128 float heightmap from the VAE decoder (Image.FORMAT_RF -> .r channel).
uniform sampler2D heightmap : hint_default_white, filter_linear, repeat_disable;
uniform float height_scale = 22.0;
uniform float plane_size = 240.0;

// Mountain-shaping controls (tune live in the inspector or via UI sliders).
uniform float sharpness = 1.6;   // >1: harder ridges, deeper grooves. 1.0 = raw VAE output.
uniform float ridge_amt = 0.35;  // dendritic crest strength (0 = off).
uniform float groove_amt = 0.45; // tree-like valley carving depth (0 = off).
uniform float ridge_scale = 44.0; // spatial frequency of the ridge/valley network.
uniform float warp_amt = 1.2;    // how much ridges curl and meander.
uniform float crest_thresh = 0.30; // ridged-field level above which crests rise.
uniform float groove_thresh = 0.34; // ridged-field level below which valleys carve.

varying float v_height;

float hash(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float vnoise(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
			mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

// Domain-warped ridged multifractal: sharp dendritic crests that curl and
// meander like real erosion ridges, instead of running in straight lines.
float ridged(vec2 p) {
	vec2 q = vec2(vnoise(p + vec2(0.0, 0.0)), vnoise(p + vec2(5.2, 1.3)));
	vec2 w = p + warp_amt * (q - 0.5) * 2.0;
	float sum = 0.0;
	float amp = 0.55;
	vec2 pp = w;
	for (int i = 0; i < 4; i++) {
		float n = vnoise(pp);
		n = 1.0 - abs(2.0 * n - 1.0); // fold into a sharp crest
		n = n * n;
		sum += n * amp;
		pp = pp * 2.03 + vec2(1.7, 9.2);
		amp *= 0.5;
	}
	return clamp(sum, 0.0, 1.0);
}

// Full terrain height at a UV: VAE base -> contrast curve -> crests up, tree-valleys carved down.
float terrain_h(vec2 uv) {
	float h = clamp(texture(heightmap, uv).r, 0.0, 1.0);
	float base = pow(h, sharpness);
	float r = ridged(uv * ridge_scale);
	// Crests thin with altitude; the low regions of the same warped field become
	// branching tree-like valleys, carved slightly deeper than crests rise.
	float thin = mix(1.0, 2.6, base);
	float crest = max(pow(r, thin) - crest_thresh, 0.0) * ridge_amt;
	float groove = max(groove_thresh - r, 0.0) * groove_amt;
	float mask = smoothstep(0.02, 0.3, base); // keep water flat
	return base + (crest - groove) * mask;
}

void vertex() {
	float e = 0.002;
	float hC = terrain_h(UV);
	float hR = terrain_h(UV + vec2(e, 0.0));
	float hL = terrain_h(UV - vec2(e, 0.0));
	float hD = terrain_h(UV + vec2(0.0, e));
	float hU = terrain_h(UV - vec2(0.0, e));
	v_height = clamp(hC, 0.0, 1.0);
	VERTEX.y += hC * height_scale;

	// Finite-difference normal: delta in world coordinates
	float world_step = e * plane_size;
	float dhdx = (hR - hL) * height_scale / (2.0 * world_step);
	float dhdz = (hD - hU) * height_scale / (2.0 * world_step);
	NORMAL = normalize(vec3(-dhdx, 1.0, -dhdz));
}

void fragment() {
	float h = v_height;
	// Hypsometric tint: vivid water -> lush grass -> warm rock -> snow.
	vec3 col = mix(vec3(0.12, 0.42, 0.88), vec3(0.20, 0.68, 0.30), smoothstep(0.05, 0.25, h));
	col = mix(col, vec3(0.66, 0.48, 0.30), smoothstep(0.25, 0.55, h));
	col = mix(col, vec3(0.95, 0.95, 1.0), smoothstep(0.55, 0.80, h));

	ALBEDO = col;
	ROUGHNESS = 0.85;
}
