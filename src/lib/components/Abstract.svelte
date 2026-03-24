<section>
	<div class="mb-2 text-[10px] font-bold tracking-[0.15em] text-primary uppercase">Abstract</div>
	<p class="border-l-2 border-primary pl-5 text-[0.93rem] leading-[1.88] text-foreground/80">
		As vision-language models are deployed at scale, understanding their internal mechanisms becomes
		increasingly critical. Existing interpretability methods predominantly rely on activations,
		making them dataset-dependent, vulnerable to data bias, and often restricted to coarse
		head-level explanations. We introduce
		<strong>SITH</strong> (<strong>S</strong>emantic <strong>I</strong>nspection of
		<strong>T</strong>ransformer <strong>H</strong>eads), a fully data-free, training-free framework
		that directly analyzes CLIP&rsquo;s vision transformer in <em>weight space</em>. For each
		attention head, we decompose its value-output matrix into singular vectors and interpret each
		one via <strong>COMP</strong> (<strong>C</strong>oherent <strong>O</strong>rthogonal
		<strong>M</strong>atching <strong>P</strong>ursuit), a new algorithm that explains them as
		sparse, semantically coherent combinations of human-interpretable concepts. We show that SITH
		yields monosemantic, faithful intra-head explanations, validated through reconstruction fidelity
		and interpretability experiments. This enables precise, interpretable weight-space model edits
		that amplify or suppress specific concepts, improving downstream performance without retraining.
		Furthermore, SITH reveals how fine-tuning primarily reweights a stable semantic basis rather
		than learning entirely new features.
	</p>
</section>
