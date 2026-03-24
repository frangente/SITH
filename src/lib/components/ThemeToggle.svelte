<script lang="ts">
	import { toggleMode } from 'mode-watcher';
	import SunIcon from '~icons/heroicons/sun';
	import MoonIcon from '~icons/heroicons/moon';
</script>

<button
	onclick={toggleMode}
	aria-label="Toggle light/dark theme"
	class="fixed top-[1.1rem] right-[1.4rem] z-200 flex size-8.5
	       cursor-pointer items-center justify-center rounded-full
	       border border-input bg-card text-muted-foreground
	       transition-colors duration-150
	       hover:border-primary-border hover:text-primary"
>
	<MoonIcon class="dark:hidden" />
	<SunIcon class="hidden dark:block" />
</button>
