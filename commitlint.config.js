export default {
	extends: ['@commitlint/config-conventional'],
	rules: {
		'type-enum': [2, 'always', ['feat', 'fix', 'refactor', 'docs', 'style', 'chore', 'test']],
		'scope-enum': [2, 'always', ['core', 'cli', 'web', 'config', 'deps']],
		'scope-empty': [2, 'never'],
		'subject-empty': [2, 'never'],
		'subject-case': [2, 'always', 'lower-case'],
		'subject-full-stop': [2, 'never', '.'],
		'header-max-length': [2, 'always', 100],
		'body-max-line-length': [2, 'always', 100],
	},
};
