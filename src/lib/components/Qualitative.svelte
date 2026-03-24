<script lang="ts">
	import explanations from '$lib/assets/explanations.json';

	type Concept = { name: string; score: number };
	type Side = { label: string; concepts: Concept[] };
	type SV = { index: number; left?: Side; right?: Side };
	type Head = { tab_label: string; description: string; svs?: SV[] };

	let groups = $derived.by(() => {
		// eslint-disable-next-line svelte/prefer-svelte-reactivity
		const map = new Map<string, Head[]>();
		for (const head of explanations) {
			if (!map.has(head.model)) map.set(head.model, []);
			map.get(head.model)!.push(head);
		}
		return Array.from(map.entries());
	});

	// Selection state — one active head per model group, one active SV per (model, head) pair
	let selectedModel = $state(0);
	let headSelections = $state<Record<number, number>>({});
	let svSelections = $state<Record<string, number>>({});

	// Derived active data
	let allGroups = $derived(groups.map(([, h]) => h));
	let modelNames = $derived(groups.map(([m]) => m));
	let currentHeads = $derived(allGroups[selectedModel] ?? []);
	let currentHeadIdx = $derived(headSelections[selectedModel] ?? 0);
	let currentHeadData = $derived(currentHeads[currentHeadIdx] ?? null);
	let currentSVIdx = $derived(svSelections[`${selectedModel}-${currentHeadIdx}`] ?? 0);
	let currentSV = $derived(currentHeadData?.svs?.[currentSVIdx] ?? null);

	// Render concept names — \censor{word} → blurred span (hover to reveal)
	function renderName(raw: string) {
		return raw.replace(
			/\\censor\{([^}]+)\}/g,
			(_, w) => `<span class="c-cens" title="hover to reveal">${w}</span>`
		);
	}
</script>

{#snippet conceptList(concepts: Concept[])}
	{@const maxScore = Math.max(...concepts.map((c) => c.score))}
	<div class="c-list">
		{#each concepts as c (c.name)}
			{@const pct = Math.round((c.score / maxScore) * 100)}
			<div class="c-row">
				<span class="c-name">{@html renderName(c.name)}</span>
				<div class="c-bar-track"><div class="c-bar" style="width:{pct}%"></div></div>
				<span class="c-score">{c.score.toFixed(3)}</span>
			</div>
		{/each}
	</div>
{/snippet}

{#snippet vectorCard(sv: SV)}
	{@const hasLeft = (sv.left?.concepts?.length ?? 0) > 0}
	{@const hasRight = (sv.right?.concepts?.length ?? 0) > 0}
	{@const single = !hasLeft || !hasRight}
	<div class="vector-card" class:vector-card--single={single}>
		{#if hasLeft}
			<div class="v-side">
				<div class="v-dir">← Left direction</div>
				<div class="v-title">{sv.left!.label}</div>
				{@render conceptList(sv.left!.concepts)}
			</div>
		{/if}
		{#if !single}<div class="v-divider"></div>{/if}
		{#if hasRight}
			<div class="v-side">
				<div class="v-dir">
					{single ? 'Concept decomposition (right singular vector)' : 'Right direction →'}
				</div>
				<div class="v-title">{sv.right!.label}</div>
				{@render conceptList(sv.right!.concepts)}
			</div>
		{/if}
	</div>
{/snippet}

<div class="mb-2 text-[10px] font-bold tracking-[0.15em] text-primary uppercase">
	Qualitative Results
</div>
<h2 class="mb-[1.4rem] font-serif text-4xl leading-tight font-normal tracking-[-0.01em]">
	What Do Individual Heads <em class="text-primary italic">Encode</em>?
</h2>

<p class="mb-4 text-[0.88rem] leading-[1.8] text-muted-foreground">
	By considering the first few singular vectors of each Value-Output matrix, which represent the
	dominant directions of information flow through each head, we can identify the primary functional
	role of each head. Exploring these singular vectors reveals striking semantic structure: many
	individual heads naturally
	<strong class="font-medium text-foreground/80">specialize in distinct, coherent themes</strong>,
	such as locations, colors, or materials. Furthermore, we observe the same phenomenon across all
	examined CLIP-trained models, suggesting that this semantic specialization is a
	<strong class="font-medium text-foreground/80">universal property</strong>
	of CLIP-trained models. As shown below, the exact same functionally specialized heads emerge across
	vastly different model capacities and architectures, from ViT-B/32 to MobileCLIP.
</p>

<!-- Model-level tabs -->
<div class="model-tabs" role="tablist">
	{#each modelNames as name, i (name)}
		<button
			class="model-tab"
			class:active={selectedModel === i}
			role="tab"
			onclick={() => {
				selectedModel = i;
			}}>{name}</button
		>
	{/each}
</div>

<!-- Head-level tabs -->
<div class="head-tabs" role="tablist">
	{#each currentHeads as head, hi (head.tab_label)}
		<button
			class="head-tab"
			class:active={currentHeadIdx === hi}
			role="tab"
			onclick={() => {
				headSelections[selectedModel] = hi;
			}}>{head.tab_label}</button
		>
	{/each}
</div>

{#if currentHeadData}
	<p class="head-meta"><strong>{currentHeadData.description}</strong></p>

	{#if currentHeadData.svs && currentHeadData.svs.length > 0}
		<!-- SV selector pills -->
		<div class="sv-tabs" role="tablist">
			{#each currentHeadData.svs as sv, si (sv.index)}
				<button
					class="sv-tab"
					class:active={currentSVIdx === si}
					onclick={() => {
						svSelections[`${selectedModel}-${currentHeadIdx}`] = si;
					}}>SV {sv.index + 1}</button
				>
			{/each}
		</div>

		{#if currentSV}
			{@render vectorCard(currentSV)}
		{/if}
	{:else}
		<p class="sv-empty">Qualitative examples for this head will be added soon.</p>
	{/if}
{/if}

<style>
	/* ── Model tabs ─────────────────────────────────────── */
	.model-tabs {
		display: flex;
		border-bottom: 1px solid var(--input);
		margin-bottom: 1.4rem;
		overflow-x: auto;
		scrollbar-width: none;
	}
	.model-tabs::-webkit-scrollbar {
		display: none;
	}

	.model-tab {
		padding: 8px 18px;
		font-size: 0.82rem;
		font-weight: 600;
		font-family: var(--font-sans);
		color: var(--muted-foreground);
		background: none;
		border: none;
		border-bottom: 2px solid transparent;
		margin-bottom: -1px;
		cursor: pointer;
		transition:
			color 0.13s,
			border-color 0.13s;
		white-space: nowrap;
	}
	.model-tab:hover {
		color: var(--foreground);
	}
	.model-tab.active {
		color: var(--primary);
		border-bottom-color: var(--primary);
	}

	/* ── Head tabs ──────────────────────────────────────── */
	.head-tabs {
		display: flex;
		border-bottom: 1px solid var(--input);
		margin-bottom: 1.4rem;
	}

	.head-tab {
		padding: 8px 18px;
		font-size: 0.8rem;
		font-weight: 500;
		font-family: var(--font-sans);
		color: var(--muted-foreground);
		background: none;
		border: none;
		border-bottom: 2px solid transparent;
		margin-bottom: -1px;
		cursor: pointer;
		transition:
			color 0.13s,
			border-color 0.13s;
	}
	.head-tab:hover {
		color: var(--foreground);
	}
	.head-tab.active {
		color: var(--primary);
		border-bottom-color: var(--primary);
	}

	/* ── Head meta ──────────────────────────────────────── */
	.head-meta {
		font-size: 0.82rem;
		color: var(--muted-foreground);
		font-style: italic;
		margin-bottom: 1.2rem;
	}
	.head-meta strong {
		color: var(--foreground);
		font-style: normal;
	}

	/* ── SV pills ───────────────────────────────────────── */
	.sv-tabs {
		display: flex;
		gap: 5px;
		flex-wrap: wrap;
		margin-bottom: 1.1rem;
	}

	.sv-tab {
		padding: 4px 11px;
		font-size: 0.73rem;
		font-weight: 500;
		font-family: var(--font-sans);
		color: var(--muted-foreground);
		background: var(--card);
		border: 1px solid var(--border);
		border-radius: 2px;
		cursor: pointer;
		transition: all 0.12s;
	}
	.sv-tab:hover {
		color: var(--foreground);
		border-color: var(--input);
	}
	.sv-tab.active {
		color: var(--primary);
		background: var(--primary-subtle);
		border-color: var(--primary-border);
	}

	/* ── Vector card ────────────────────────────────────── */
	.vector-card {
		background: var(--card);
		border: 1px solid var(--input);
		border-radius: 5px;
		overflow: hidden;
		display: grid;
		grid-template-columns: 1fr 1px 1fr;
	}
	.vector-card--single {
		grid-template-columns: 1fr;
	}

	.v-side {
		padding: 1.4rem 1.5rem;
	}

	.v-divider {
		background: var(--input);
		position: relative;
	}
	.v-divider::after {
		content: '↔';
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		font-size: 12px;
		color: var(--accent-foreground);
		background: var(--card);
		padding: 3px 0;
		line-height: 1;
	}

	.v-dir {
		font-size: 9px;
		font-weight: 700;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		color: var(--muted-foreground);
		margin-bottom: 5px;
	}

	.v-title {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 1.05rem;
		font-weight: 400;
		color: var(--foreground);
		margin-bottom: 1rem;
	}

	/* ── Concept list ───────────────────────────────────── */
	.c-list {
		display: flex;
		flex-direction: column;
		gap: 7px;
	}

	.c-row {
		display: grid;
		grid-template-columns: 1fr 72px 34px;
		align-items: center;
		gap: 8px;
	}

	.c-name {
		font-size: 0.78rem;
		color: color-mix(in srgb, var(--foreground) 80%, transparent);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.c-bar-track {
		height: 2px;
		background: var(--input);
		border-radius: 1px;
		overflow: hidden;
	}
	.c-bar {
		height: 100%;
		background: var(--primary);
		border-radius: 1px;
		opacity: 0.75;
	}

	.c-score {
		font-size: 0.68rem;
		color: var(--muted-foreground);
		text-align: right;
		font-variant-numeric: tabular-nums;
		font-family: 'Courier New', monospace;
	}

	/* .c-cens is injected via {@html}, so it needs :global */
	:global(.c-cens) {
		filter: blur(4px);
		cursor: pointer;
		transition: filter 0.2s;
		user-select: none;
	}
	:global(.c-cens:hover) {
		filter: none;
	}

	/* ── Empty state ────────────────────────────────────── */
	.sv-empty {
		font-size: 0.83rem;
		color: var(--muted-foreground);
		font-style: italic;
		padding: 1.5rem 0;
	}

	/* ── Responsive ─────────────────────────────────────── */
	@media (max-width: 640px) {
		.vector-card {
			grid-template-columns: 1fr;
		}
		.v-divider {
			display: none;
		}
	}
</style>
