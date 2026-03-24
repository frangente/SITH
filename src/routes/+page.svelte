<script lang="ts">
	import Abstract from '$lib/components/Abstract.svelte';
	import BibTeX from '$lib/components/BibTeX.svelte';
	import Experiments from '$lib/components/Experiments.svelte';
	import Footer from '$lib/components/Footer.svelte';
	import Header from '$lib/components/Header.svelte';
	import Method from '$lib/components/Method.svelte';
	import Qualitative from '$lib/components/Qualitative.svelte';
	import ThemeToggle from '$lib/components/ThemeToggle.svelte';
</script>

<svelte:head>
	<title>SITH — From Weights to Concepts</title>
	<meta
		name="description"
		content="SITH: data-free, training-free interpretability of CLIP's vision transformer via SVD. CVPR 2026."
	/>
</svelte:head>

<ThemeToggle />
<Header />

<main class="mx-auto max-w-220 px-8">
	<div class="border-b border-border py-16"><Abstract /></div>
	<div class="border-b border-border py-16"><Method /></div>
	<div class="border-b border-border py-16"><Qualitative /></div>
	<div class="border-b border-border py-16"><Experiments /></div>
	<div class="py-16"><BibTeX /></div>
</main>

<Footer />
