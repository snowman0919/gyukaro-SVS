# SVS Phase 0 handoff

- Branch: `codex/svs-01-research-evidence`
- Base commit: `1e72b4a73ec1836651fac1f5df503b1b57213b93`
- Implementation HEAD: `16c6910b288d933f2193a0edbc61869579124551`
- Commits: `16c6910 feat(research): normalize SVS evidence status`; this handoff document is the closing documentation commit.
- Tests: `141 passed, 13 warnings`
- Dataset: `PASS recordings=132 sequential=106..237 pcm=48k_mono corrupt=0`
- Evidence check: `PASS models=9 evidence=8`
- Known limitations: Korean lexical status is deliberately `foundation_ko_lexical_unverified`; historical untouched DiffSinger YAML files still contain machine-specific evidence paths; no runtime, model, package, or OpenUtau behavior changed.
- Local evidence: repository-level `data/cache`, `data/processed`, and `data/external/{raw,work}` are linked into this ignored worktree; no binary evidence is committed.
- Next branch: `codex/svs-02-experiment-framework`, based on the closing commit of this branch.
