<script lang="ts">
	import MethodImage from '$lib/assets/method.png';

	let modalOpen = $state(false);

	const steps = [
		{
			n: 1,
			title: 'Decompose W<sub>VO</sub> via SVD',
			text: `For each attention head, SITH isolates its value-output weight matrix
				<strong>W</strong><sub>VO</sub> and factorizes it into <em>r</em> singular vectors
				{ <strong>v<sub>j</sub></strong> } via SVD. These vectors capture the head\u2019s dominant
				computational directions \u2014 directly from weights, requiring no images,
				activations, or labels.`
		},
		{
			n: 2,
			title: 'Explain each vector with COMP',
			text: `Each singular vector is decomposed by <strong>COMP</strong>
				(<strong>C</strong>oherent <strong>O</strong>rthogonal <strong>M</strong>atching
				<strong>P</strong>ursuit) into a sparse, non-negative combination of concept embeddings
				drawn from a textual dictionary &Gamma; (e.g., ConceptNet). Unlike standard matching
				pursuit, COMP adds a coherence term that favours concepts semantically consistent with
				those already selected \u2014 producing explanations that read as meaningful groups,
				not unrelated word lists.`
		}
	];
</script>

<div class="mb-2 text-[10px] font-bold tracking-[0.15em] text-primary uppercase">Method</div>
<h2 class="mb-[1.4rem] font-serif text-4xl leading-tight font-normal tracking-[-0.01em]">
	How SITH Works
</h2>

<div class="mb-8 flex flex-col gap-5">
	{#each steps as step (step.n)}
		<div class="grid grid-cols-[36px_1fr] items-start gap-4">
			<div
				class="mt-0.5 flex size-8 shrink-0 items-center
				       justify-center rounded-full border border-primary-border bg-primary-subtle text-[0.75rem]
				       font-bold text-primary"
			>
				{step.n}
			</div>
			<div>
				<div class="mb-1 font-serif text-[1.05rem] leading-[1.3] font-normal text-foreground">
					<!-- eslint-disable-next-line svelte/no-at-html-tags -->
					{@html step.title}
				</div>
				<p class="text-[0.85rem] leading-[1.78] text-muted-foreground">
					<!-- eslint-disable-next-line svelte/no-at-html-tags -->
					{@html step.text}
				</p>
			</div>
		</div>
	{/each}
</div>

<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<figure>
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<img
		src={MethodImage}
		alt="Overview of SITH and COMP"
		loading="lazy"
		class="block w-full cursor-zoom-in rounded-[5px] border border-input"
		onclick={() => (modalOpen = true)}
	/>
	<figcaption class="mt-2.5 text-[0.78rem] leading-[1.65] text-muted-foreground">
		<strong class="font-medium text-foreground/80">Figure.</strong>
		(<em>left</em>) SITH isolates <em>W</em><sub>VO</sub> and applies SVD, yielding singular vectors
		{'{<em>v<sub>j</sub></em>}'}. Given concept pool &Gamma;, COMP explains each vector as a sparse
		coherent combination of concept embeddings. (<em>right</em>) COMP iteratively builds the concept
		set by selecting, at each step, the concept most similar to the current residual
		<em>and</em> semantically consistent with the concepts already chosen.
	</figcaption>
</figure>

<!-- Image zoom modal -->
{#if modalOpen}
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		aria-modal="true"
		aria-label="Enlarged view of method overview"
		class="fixed inset-0 z-1000 flex cursor-pointer items-center justify-center bg-black/90"
		onclick={() => (modalOpen = false)}
	>
		<img
			src={MethodImage}
			alt="Overview of SITH and COMP"
			class="max-h-[90%] max-w-[90%] object-contain"
		/>
		<button
			class="absolute top-4 right-8 cursor-pointer text-4xl leading-none font-bold
			       text-[#f1f1f1] transition-colors hover:text-[#bbb]"
			onclick={() => (modalOpen = false)}
			aria-label="Close"
		>
			&times;
		</button>
	</div>
{/if}
