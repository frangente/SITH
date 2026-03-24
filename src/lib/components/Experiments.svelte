<script lang="ts">
	const experiments = [
		{
			n: 1,
			tag: 'Spurious Correlations',
			title: 'Background-Invariant Bird Classification',
			desc: `On Waterbirds, CLIP can exploit background habitat cues (water vs. land) rather
				than bird appearance to classify the two classes (waterbird vs. landbird). Thus,
				we use SITH to identify and suppress the singular vectors encoding background information,
				without any additional training or labels. This simple edit improves worst-group accuracy
				by a wide margin (+22.7pp), outperforming TextSpan, a recent method that removes spurious
				features by ablating entire heads, showing that more fine-grained editing can be more effective.`,
			metric: '+22.7pp',
			metricSub: 'worst-group accuracy\n47.9% \u2192 70.6%'
		},
		{
			n: 2,
			tag: 'NSFW Removal',
			title: 'Safety-Aware Retrieval Without Retraining',
			desc: `CLIP can encode and retrieve unsafe visual content, thus requiring costly retraining to
				meet safety standards. Instead, we use SITH to identify the singular vectors associated with
				nudity and/or violence, and suppress them to create a safer model without any training or
				labeled data. On the ViSU benchmark, the edited model improves retrieval recall on unsafe
				visual queries, in particular for unsafe visual queries.`,
			metric: '+1.5pp',
			metricSub: 'for I*\u2192T queries\n38.9% \u2192 40.4%'
		},
		{
			n: 3,
			tag: 'Classification Performance',
			title: 'Task-Aware Singular Value Amplification',
			desc: `For a target classification task, we use SITH to measure the alignment between the semantic
				content of each singular vector and task-relevant concepts. We then edit the model by rescaling
				proportionally the singular values, so as to amplify relevant directions and downweight
				irrelevant ones. This simple edit yields consistent gains across Flowers\u00a0102, FGVC-Aircraft,
				and DTD, without any training, labeled data, or gradient computation.`,
			metric: '+1.0pp',
			metricSub: 'Flowers 102\n76.5% \u2192 77.5%'
		}
	];
</script>

<div class="mb-2 text-[10px] font-bold tracking-[0.15em] text-primary uppercase">Applications</div>
<h2 class="mb-[1.4rem] font-serif text-4xl leading-tight font-normal tracking-[-0.01em]">
	Interpretable <em class="text-primary italic">Model Editing</em> with SITH
</h2>
<p class="mb-6 text-[0.88rem] leading-[1.8] text-muted-foreground">
	Because SITH operates directly on weights, its decompositions are immediately actionable. By
	scaling the singular values associated with identified concept directions, we can surgically
	modify CLIP&rsquo;s behavior &mdash; suppressing unwanted features or amplifying task-relevant
	ones &mdash; with no training, no labeled data, and no gradient computation.
</p>

<div class="edit-stack">
	{#each experiments as exp (exp.n)}
		<div class="edit-item">
			<div class="edit-n">{exp.n}</div>
			<div>
				<div class="edit-tag">{exp.tag}</div>
				<div class="edit-title">{exp.title}</div>
				<p class="edit-desc">{exp.desc}</p>
			</div>
			<div class="edit-metric">
				<div class="metric-val">{exp.metric}</div>
				<div class="metric-sub">
					{#each exp.metricSub.split('\n') as line, i (i)}
						{#if i > 0}<br />{/if}{line}
					{/each}
				</div>
			</div>
		</div>
	{/each}
</div>

<style>
	.edit-stack {
		display: flex;
		flex-direction: column;
		border: 1px solid var(--input);
		border-radius: 5px;
		overflow: hidden;
	}

	.edit-item {
		display: grid;
		grid-template-columns: 44px 1fr auto;
		gap: 1.4rem;
		padding: 1.75rem;
		background: var(--card);
		align-items: start;
	}
	.edit-item + .edit-item {
		border-top: 1px solid var(--border);
	}

	.edit-n {
		width: 32px;
		height: 32px;
		border-radius: 50%;
		border: 1px solid var(--primary-border);
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 0.72rem;
		font-weight: 600;
		color: var(--primary);
		margin-top: 3px;
		flex-shrink: 0;
	}

	.edit-tag {
		font-size: 9.5px;
		font-weight: 700;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		color: var(--muted-foreground);
		margin-bottom: 3px;
	}

	.edit-title {
		font-family: var(--font-serif);
		font-size: 1.1rem;
		font-weight: 400;
		color: var(--foreground);
		margin-bottom: 5px;
		line-height: 1.3;
	}

	.edit-desc {
		font-size: 0.82rem;
		color: var(--muted-foreground);
		line-height: 1.75;
	}

	.edit-metric {
		text-align: right;
		flex-shrink: 0;
		padding-top: 2px;
	}

	.metric-val {
		font-family: var(--font-serif);
		font-size: 1.65rem;
		font-weight: 300;
		color: var(--primary);
		line-height: 1;
		white-space: nowrap;
	}

	.metric-sub {
		font-size: 0.67rem;
		color: var(--muted-foreground);
		margin-top: 4px;
		line-height: 1.5;
		white-space: nowrap;
	}

	@media (max-width: 640px) {
		.edit-item {
			grid-template-columns: 44px 1fr;
		}
		.edit-metric {
			grid-column: 2;
			text-align: left;
		}
	}
</style>
