"""Unit test to verify all modules can be imported and config instantiates."""

def test_module_imports():
    """Asserts that every module under rem/ can be imported and config loads."""
    # Import config first
    from rem.config import Settings
    settings = Settings()
    assert settings is not None
    assert settings.litellm_port == 4000

    # Import other modules to verify stubs/syntaxes
    import rem
    import rem.npu_client
    import rem.platform.gate
    import rem.platform.probe_embeddings
    import rem.platform.discover_sysfs
    import rem.memory.tiers
    import rem.memory.facts_ledger
    import rem.memory.compactor
    import rem.memory.assembler
    import rem.memory.export
    import rem.wiki.schema
    import rem.wiki.vault
    import rem.wiki.compiler
    import rem.wiki.recall
    import rem.scheduler.gauge
    import rem.scheduler.admission
    import rem.scheduler.queue
    import rem.scheduler.dispatcher
    import rem.scheduler.telemetry

    assert rem.__version__ == "0.1.0"
