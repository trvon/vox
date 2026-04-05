# Changelog

## [0.3.0](https://github.com/trvon/vox/compare/v0.2.0...v0.3.0) (2026-04-05)


### Features

* **config:** add seamless Kokoro model selection with auto-download and hot swap ([94044e7](https://github.com/trvon/vox/commit/94044e73a2d9417c610d831d262092d14fb77418))
* quality of life improvements ([14ef7d5](https://github.com/trvon/vox/commit/14ef7d5b5e9b19886727823ffc38c06b8a1dc832))
* **server:** add ordered TTS queue tools ([0593fa3](https://github.com/trvon/vox/commit/0593fa3623be54883a5f6fa7b7fe8ab97bd28a9d))
* updating cargo.lock ([e616bfb](https://github.com/trvon/vox/commit/e616bfb2578b83c52afb58be56854c6d9745ba27))
* **vox:** add daemon-backed tool CLI and extract shared voice runtime ([cd76993](https://github.com/trvon/vox/commit/cd7699399452796385d2992dacdd73f446d74366))


### Bug Fixes

* linux build fixes ([fcfe811](https://github.com/trvon/vox/commit/fcfe81111b73145fb82870bbe950840ce00a13b8))
* **models:** stream archive extraction to avoid download OOM ([2894766](https://github.com/trvon/vox/commit/2894766db5cb4b1e30a351313dc1c1b200018f1c))

## [0.2.0](https://github.com/trvon/vox/compare/v0.1.0...v0.2.0) (2026-02-09)


### Features

* removing timout for MCP, updating ci ([611044b](https://github.com/trvon/vox/commit/611044b099e32a034b45b3a2a56879f5fc0d02f0))

## 0.1.0 (2026-02-08)


### Features

* hello world, heres the files ([8123e9a](https://github.com/trvon/vox/commit/8123e9adcd0463e819215dfc565bc07a7ccd0c55))
* improved pipeline and quality ([6df2831](https://github.com/trvon/vox/commit/6df283142625a2381c27c2d06fe59a3ca0577912))
* improving spsed ([092c731](https://github.com/trvon/vox/commit/092c73193c651f3fbd65998304802531d29145b8))
* initial vox MCP server implementation ([88f2166](https://github.com/trvon/vox/commit/88f2166c7bac59cb87ea0e4c43bc7cf5618b7c72))
* mcp imrpovements, warm period for speed, calibrate for setup optimizations ([dcd5f85](https://github.com/trvon/vox/commit/dcd5f85a9844edd248c6d3888049df884b7978a5))
* switch STT from Moonshine Tiny to Moonshine Base for better accuracy ([9e88da0](https://github.com/trvon/vox/commit/9e88da0fc0d4869aa4239167b084e958a5e77f1c))


### Bug Fixes

* adding mold to ci ([4e8a562](https://github.com/trvon/vox/commit/4e8a562cd5f1730065e4590541f926b70839f716))
* adding setup.sh but honestly just cargo install ([2c8e2b6](https://github.com/trvon/vox/commit/2c8e2b674f03e86ffa3b4170fc44dfde9bebe3cd))
* ci improvements ([b821c28](https://github.com/trvon/vox/commit/b821c282c5e81598d89a6467f2ea51be91fcc6fa))
* fixing cargo config ([b8c3b66](https://github.com/trvon/vox/commit/b8c3b669849c96798ae55f9d8d5e7826a9c33cec))
* more ci fixes ([8be40b7](https://github.com/trvon/vox/commit/8be40b771ac22356cf1c24d3ba2876f1d36b3cfd))
* running clippy ([388d27f](https://github.com/trvon/vox/commit/388d27fa8e9885dd38f28f1c01231f6b46f7e352))
