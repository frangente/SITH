<script lang="ts">
	import CopyIcon from '~icons/heroicons/clipboard-document';
	import TickIcon from '~icons/heroicons/check';

	const bibtex = `@inproceedings{gentile2026sith,
	title        = {From Weights to Concepts: Data-Free Interpretability of {CLIP} via Singular Vector Decomposition},
	author       = {Gentile, Francesco and Dall'Asen, Nicola and Tonini, Francesco and Mancini, Massimiliano and Vaquero, Lorenzo and Ricci, Elisa},
	year         = 2026,
	booktitle    = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}
}`;

	let copied = $state(false);

	async function copy() {
		await navigator.clipboard.writeText(bibtex);
		copied = true;
		setTimeout(() => (copied = false), 2000);
	}
</script>

<div class="mb-2 text-[10px] font-bold tracking-[0.15em] text-primary uppercase">Citation</div>
<h2 class="mb-4 font-serif leading-tight font-normal tracking-[-0.01em]">BibTeX</h2>

<div class="overflow-hidden rounded-[5px] border border-input">
	<div class="flex items-center justify-between border-b border-border bg-card px-3.5 py-1.75">
		<span class="text-[9.5px] font-bold tracking-[0.12em] text-muted-foreground uppercase">
			BibTeX
		</span>
		<button
			onclick={copy}
			class="inline-flex cursor-pointer items-center gap-1.25 rounded-xs border
			       border-border bg-none px-2.25 py-0.75 text-[0.71rem] font-medium text-muted-foreground
			       transition-colors duration-150 hover:border-primary-border hover:text-primary"
		>
			{#if copied}
				<TickIcon class="h-3.5 w-3.5" />
				Copied
			{:else}
				<CopyIcon class="h-3.5 w-3.5" />
				Copy
			{/if}
		</button>
	</div>
	<pre
		class="overflow-x-auto bg-card px-[1.4rem] py-[1.2rem] font-mono text-[0.77rem]
		       leading-[1.75] whitespace-pre text-muted-foreground">{bibtex}</pre>
</div>
