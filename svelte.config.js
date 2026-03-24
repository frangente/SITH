import adapter from '@sveltejs/adapter-static';

// When built by GitHub Actions, BASE_PATH is set to the repo subpath (e.g. /SITH).
// Locally it stays empty so `npm run dev` works without any prefix.
const base = process.env.BASE_PATH ?? '';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	kit: {
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			fallback: undefined,
			precompress: false,
			strict: true
		}),
		paths: { base }
	},
	vitePlugin: {
		dynamicCompileOptions: ({ filename }) =>
			filename.includes('node_modules') ? undefined : { runes: true }
	}
};

export default config;
