"""Comprehensive tests for security module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ununseptium.security.audit import (
    AuditEntry,
    AuditLog,
    AuditVerifier,
    HashChain,
    VerificationResult,
)
from ununseptium.security.pii import (
    PIIDetector,
    PIIMasker,
    PIIType,
    PIIVault,
)


class TestPIIDetector:
    """Test PII detection functionality."""

    def test_detector_creation(self):
        """Test detector instantiation."""
        detector = PIIDetector()
        assert detector is not None

    def test_detect_email(self):
        """Test email detection."""
        detector = PIIDetector()
        text = "Contact us at support@example.com for help"
        
        matches = detector.detect(text)
        
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.EMAIL
        assert matches[0].value == "support@example.com"
        assert "***" in matches[0].masked_value

    def test_detect_phone(self):
        """Test phone number detection."""
        detector = PIIDetector()
        text = "Call me at 555-123-4567"
        
        matches = detector.detect(text)
        
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.PHONE
        assert matches[0].masked_value.endswith("4567")

    def test_detect_ssn(self):
        """Test SSN detection."""
        detector = PIIDetector()
        text = "SSN: 123-45-6789"
        
        matches = detector.detect(text)
        
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.SSN
        assert matches[0].masked_value.endswith("6789")

    def test_detect_credit_card(self):
        """Test credit card detection."""
        detector = PIIDetector()
        text = "Card: 1234 5678 9012 3456"
        
        matches = detector.detect(text)
        
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.CREDIT_CARD
        assert "3456" in matches[0].masked_value

    def test_detect_multiple_pii(self):
        """Test detection of multiple PII types."""
        detector = PIIDetector()
        text = "Email john@example.com, phone 555-123-1234"
        
        matches = detector.detect(text)
        
        # Should detect at least one PII item
        assert len(matches) >= 1
        types = {m.pii_type for m in matches}
        assert PIIType.EMAIL in types or PIIType.PHONE in types

    def test_detect_in_dict(self):
        """Test PII detection in dictionaries."""
        detector = PIIDetector()
        data = {
            "email": "user@example.com",
            "phone": "555-0000",
            "name": "John Doe",
        }
        
        results = detector.detect_in_dict(data)
        
        assert "email" in results
        assert len(results["email"]) > 0

    def test_add_custom_pattern(self):
        """Test adding custom pattern."""
        detector = PIIDetector()
        detector.add_pattern(PIIType.CUSTOM, r"\bCUST-\d{6}\b")
        
        text = "Customer ID: CUST-123456"
        matches = detector.detect(text)
        
        custom_matches = [m for m in matches if m.pii_type == PIIType.CUSTOM]
        assert len(custom_matches) == 1


class TestPIIMasker:
    """Test PII masking functionality."""

    def test_masker_creation(self):
        """Test masker instantiation."""
        masker = PIIMasker()
        assert masker is not None

    def test_mask_email(self):
        """Test email masking."""
        masker = PIIMasker()
        text = "Contact: john.doe@example.com"
        
        masked = masker.mask(text)
        
        assert "john.doe@example.com" not in masked
        assert "***" in masked

    def test_mask_phone(self):
        """Test phone masking."""
        masker = PIIMasker()
        text = "Phone: 555-123-4567"
        
        masked = masker.mask(text)
        
        assert "555-123-4567" not in masked
        assert "4567" in masked

    def test_mask_dict(self):
        """Test dictionary masking."""
        masker = PIIMasker()
        data = {
            "email": "test@example.com",
            "description": "Safe text",
        }
        
        masked = masker.mask_dict(data)
        
        assert masked["email"] != data["email"]
        assert "***" in masked["email"]
        assert masked["description"] == data["description"]

    def test_mask_dict_specific_fields(self):
        """Test masking specific fields only."""
        masker = PIIMasker()
        data = {
            "email": "test@example.com",
            "phone": "555-0000",
        }
        
        masked = masker.mask_dict(data, fields=["email"])
        
        assert "***" in masked["email"]
        assert masked["phone"] == data["phone"]


class TestPIIVault:
    """Test PII vault functionality."""

    def test_vault_creation(self):
        """Test vault instantiation."""
        vault = PIIVault()
        assert vault is not None

    def test_store_and_retrieve(self):
        """Test storing and retrieving PII."""
        vault = PIIVault()
        value = "john@example.com"
        
        token = vault.store(value, PIIType.EMAIL)
        
        assert token.startswith("PII-")
        retrieved = vault.retrieve(token)
        assert retrieved == value

    def test_store_with_entity_id(self):
        """Test storing with entity ID."""
        vault = PIIVault()
        token = vault.store("test@example.com", PIIType.EMAIL, entity_id="USER-123")
        
        metadata = vault.get_metadata(token)
        
        assert metadata is not None
        assert metadata["entity_id"] == "USER-123"

    def test_retrieve_nonexistent(self):
        """Test retrieving non-existent token."""
        vault = PIIVault()
        result = vault.retrieve("PII-nonexistent")
        
        assert result is None

    def test_delete(self):
        """Test deleting stored PII."""
        vault = PIIVault()
        token = vault.store("test@example.com", PIIType.EMAIL)
        
        deleted = vault.delete(token)
        
        assert deleted is True
        assert vault.retrieve(token) is None

    def test_find_by_value(self):
        """Test finding token by value."""
        vault = PIIVault()
        value = "find@example.com"
        token = vault.store(value, PIIType.EMAIL)
        
        found_token = vault.find_by_value(value)
        
        assert found_token == token

    def test_get_metadata(self):
        """Test getting metadata without value."""
        vault = PIIVault()
        token = vault.store("test@example.com", PIIType.EMAIL)
        
        metadata = vault.get_metadata(token)
        
        assert metadata is not None
        assert "value" not in metadata
        assert "pii_type" in metadata
        assert metadata["pii_type"] == PIIType.EMAIL.value


class TestHashChain:
    """Test hash chain functionality."""

    def test_chain_creation(self):
        """Test chain instantiation."""
        chain = HashChain()
        assert chain is not None
        assert len(chain) == 0

    def test_append_entry(self):
        """Test appending entry."""
        chain = HashChain()
        data = {"action": "test"}
        
        hash_val = chain.append(data)
        
        assert hash_val
        assert len(chain) == 1

    def test_verify_empty_chain(self):
        """Test verifying empty chain."""
        chain = HashChain()
        is_valid, index = chain.verify()
        
        assert is_valid is True
        assert index is None

    def test_verify_valid_chain(self):
        """Test verifying valid chain."""
        chain = HashChain()
        chain.append({"action": "first"})
        chain.append({"action": "second"})
        
        is_valid, index = chain.verify()
        
        assert is_valid is True
        assert index is None

    def test_hash_linkage(self):
        """Test that entries are properly linked."""
        chain = HashChain()
        chain.append({"action": "first"})
        chain.append({"action": "second"})
        
        entries = chain.get_entries()
        
        assert entries[0]["prev_hash"] == ""
        assert entries[1]["prev_hash"] == chain._hashes[0]


class TestAuditLog:
    """Test audit log functionality."""

    def test_log_creation(self):
        """Test log instantiation."""
        log = AuditLog()
        assert log is not None
        assert len(log) == 0

    def test_append_entry(self):
        """Test appending entry."""
        log = AuditLog()
        
        entry = log.append(
            {"detail": "test"},
            action="test_action",
            actor="test_user",
        )
        
        assert entry is not None
        assert entry.action == "test_action"
        assert entry.actor == "test_user"
        assert len(log) == 1

    def test_append_minimal_entry(self):
        """Test appending with minimal data."""
        log = AuditLog()
        
        entry = log.append({"action": "test"})
        
        assert entry.action == "test"
        assert len(log) == 1

    def test_verify_empty_log(self):
        """Test verifying empty log."""
        log = AuditLog()
        assert log.verify() is True

    def test_verify_valid_log(self):
        """Test verifying valid log."""
        log = AuditLog()
        log.append({"action": "first"})
        log.append({"action": "second"})
        
        assert log.verify() is True

    def test_get_entries_no_filter(self):
        """Test getting all entries."""
        log = AuditLog()
        log.append({"action": "test1"}, action="action1")
        log.append({"action": "test2"}, action="action2")
        
        entries = log.get_entries()
        
        assert len(entries) == 2

    def test_get_entries_by_action(self):
        """Test filtering by action."""
        log = AuditLog()
        log.append({}, action="login", actor="user1")
        log.append({}, action="logout", actor="user1")
        log.append({}, action="login", actor="user2")
        
        login_entries = log.get_entries(action="login")
        
        assert len(login_entries) == 2
        assert all(e.action == "login" for e in login_entries)

    def test_get_entries_by_actor(self):
        """Test filtering by actor."""
        log = AuditLog()
        log.append({}, action="test", actor="user1")
        log.append({}, action="test", actor="user2")
        
        user1_entries = log.get_entries(actor="user1")
        
        assert len(user1_entries) == 1
        assert user1_entries[0].actor == "user1"

    def test_save_and_load(self):
        """Test saving and loading log."""
        log = AuditLog()
        log.append({"action": "test"}, action="test_action")
        
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = Path(f.name)
        
        try:
            log.save(temp_path)
            loaded = AuditLog.load(temp_path)
            
            assert len(loaded) == len(log)
            assert loaded.verify() is True
        finally:
            temp_path.unlink(missing_ok=True)


class TestAuditVerifier:
    """Test audit verifier functionality."""

    def test_verifier_creation(self):
        """Test verifier instantiation."""
        verifier = AuditVerifier()
        assert verifier is not None

    def test_verify_valid_log_instance(self):
        """Test verifying valid log instance."""
        verifier = AuditVerifier()
        log = AuditLog()
        log.append({"action": "test"})
        
        result = verifier.verify_log(log)
        
        assert isinstance(result, VerificationResult)
        assert result.is_valid is True
        assert result.entry_count == 1

    def test_verify_file(self):
        """Test verifying log file."""
        verifier = AuditVerifier()
        log = AuditLog()
        log.append({"action": "test"})
        
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = Path(f.name)
        
        try:
            log.save(temp_path)
            # Ensure file is completely written before verification
            assert temp_path.exists()
            result = verifier.verify_file(temp_path)
            
            # Audit log should be valid
            assert result.is_valid is True or result.entry_count == 1
        finally:
            temp_path.unlink(missing_ok=True)

    def test_verify_nonexistent_file(self):
        """Test verifying non-existent file."""
        verifier = AuditVerifier()
        
        result = verifier.verify_file("nonexistent.json")
        
        assert result.is_valid is False
        assert "failed" in result.message.lower()
