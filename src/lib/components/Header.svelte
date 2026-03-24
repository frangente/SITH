<script lang="ts">
	import { resolve } from '$app/paths';
	import GithubIcon from '~icons/mdi/github';
	import DownloadIcon from '~icons/mdi/download';
	import PaperIcon from '~icons/mdi/file-pdf';

	const authors = [
		{ name: 'Francesco Gentile', ref: '1*' },
		{ name: 'Nicola Dall\u2019Asen', ref: '1,2' },
		{ name: 'Francesco Tonini', ref: '1,3' },
		{ name: 'Massimiliano Mancini', ref: '1' },
		{ name: 'Lorenzo Vaquero', ref: '3' },
		{ name: 'Elisa Ricci', ref: '1,3' }
	];

	const affiliations = [
		{ num: 1, name: 'University of Trento' },
		{ num: 2, name: 'University of Pisa' },
		{ num: 3, name: 'Fondazione Bruno Kessler' }
	];
</script>

<header>
	<div class="hero-inner">
		<!-- Conference pill -->
		<div
			class="mb-[1.8rem] inline-flex items-center gap-1.25 rounded-xs border
			       border-primary-border bg-primary-subtle px-3.25 py-1 text-[10.5px]
			       font-semibold tracking-[0.13em] text-primary uppercase"
		>
			<svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
				<polygon
					points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"
				/>
			</svg>
			CVPR 2026
		</div>

		<!-- Title -->
		<h1
			class="mb-[0.9rem] font-serif leading-[1.15] font-normal tracking-[-0.02em] text-foreground"
		>
			From <em class="text-primary italic">Weights</em> to
			<em class="text-primary not-italic">Concepts</em><br />
			Data-Free Interpretability of CLIP<br />
			via Singular Vector Decomposition
		</h1>

		<!-- Subtitle -->
		<p
			class="mb-[2.4rem] font-serif text-[1.20rem] font-medium tracking-[0.015em] text-muted-foreground italic"
		>
			SITH &mdash; Semantic Inspection of Transformer Heads
		</p>

		<!-- Authors -->
		<div
			class="mb-1.25 flex flex-wrap justify-center gap-x-2.25 gap-y-0.75 text-[0.87rem] text-foreground/80"
		>
			{#each authors as author, i (author.name)}
				<span>{author.name}<sup class="text-[0.6em] font-bold text-primary">{author.ref}</sup></span
				>{#if i < authors.length - 1}<span class="text-accent-foreground">&middot;</span>{/if}
			{/each}
		</div>

		<!-- Affiliations -->
		<div
			class="mb-[2.4rem] flex flex-wrap justify-center gap-x-3.5 gap-y-0.75 text-[0.77rem] text-muted-foreground"
		>
			{#each affiliations as aff (aff.name)}
				<span><sup class="font-bold text-primary">{aff.num}</sup>&thinsp;{aff.name}</span>
			{/each}
		</div>

		<!-- Action links -->
		<nav class="flex flex-wrap justify-center gap-1.75">
			<a href={resolve('/')} class="btn btn-solid">
				<PaperIcon class="h-3.25 w-3.25" />
				Paper
			</a>
			<a href={resolve('/')} class="btn btn-ghost">arXiv</a>
			<a href="https://github.com/frangente/SITH" class="btn btn-ghost">
				<GithubIcon class="h-3.25 w-3.25" />
				Code
			</a>
			<a
				href="https://drive.google.com/drive/folders/1vL75AWVFsGPU-b1x6fenYdwjcAeL_OJx"
				class="btn btn-ghost"
			>
				<DownloadIcon class="h-3.25 w-3.25" />
				Data
			</a>
		</nav>
	</div>
</header>

<style>
	header {
		padding: 5.5rem 2rem 5rem;
		text-align: center;
		position: relative;
		overflow: hidden;
		border-bottom: 1px solid var(--border);
	}

	/* Orthogonal grid — echoes the SVD basis vectors */
	header::before {
		content: '';
		position: absolute;
		inset: 0;
		background-image:
			linear-gradient(var(--border) 1px, transparent 1px),
			linear-gradient(90deg, var(--border) 1px, transparent 1px);
		background-size: 52px 52px;
		mask-image: radial-gradient(ellipse 70% 80% at 50% 50%, black 0%, transparent 100%);
		pointer-events: none;
	}

	.hero-inner {
		position: relative;
		max-width: 900px;
		margin: 0 auto;
	}

	h1 {
		font-size: clamp(2.1rem, 5.5vw, 3.7rem);
	}

	/* Staggered entrance animations */
	@keyframes fadeUp {
		from {
			opacity: 0;
			transform: translateY(10px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	.hero-inner > * {
		animation: fadeUp 0.42s ease both;
	}
	.hero-inner > *:nth-child(1) {
		animation-delay: 0.04s;
	}
	.hero-inner > *:nth-child(2) {
		animation-delay: 0.1s;
	}
	.hero-inner > *:nth-child(3) {
		animation-delay: 0.16s;
	}
	.hero-inner > *:nth-child(4) {
		animation-delay: 0.21s;
	}
	.hero-inner > *:nth-child(5) {
		animation-delay: 0.26s;
	}
	.hero-inner > *:nth-child(6) {
		animation-delay: 0.31s;
	}

	/* Buttons */
	.btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 7px 17px;
		font-family: var(--font-sans);
		font-size: 0.79rem;
		font-weight: 500;
		border-radius: 3px;
		border: 1px solid transparent;
		transition: all 0.14s;
		cursor: pointer;
	}
	.btn :global(svg) {
		width: 13px;
		height: 13px;
		flex-shrink: 0;
	}

	.btn-solid {
		background: var(--primary);
		color: var(--primary-foreground);
		border-color: var(--primary);
	}
	.btn-solid:hover {
		filter: brightness(0.88);
	}

	.btn-ghost {
		background: transparent;
		color: color-mix(in srgb, var(--foreground) 80%, transparent);
		border-color: var(--input);
	}
	.btn-ghost:hover {
		color: var(--foreground);
	}
</style>
