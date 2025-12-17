"""Access control for security.

Provides role-based access control (RBAC) with
permission management.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class PermissionAction(str, Enum):
    """Standard permission actions."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    EXPORT = "export"
    ADMIN = "admin"


class Permission(BaseModel):
    """A permission grant.

    Attributes:
        resource: Resource identifier.
        action: Permitted action.
        conditions: Optional conditions for permission.
    """

    resource: str
    action: PermissionAction
    conditions: dict[str, Any] = Field(default_factory=dict)

    def matches(
        self,
        resource: str,
        action: PermissionAction,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Check if permission matches a request.

        Args:
            resource: Requested resource.
            action: Requested action.
            context: Request context for condition evaluation.

        Returns:
            True if permission allows the request.
        """
        # Check resource match (supports wildcards)
        if self.resource != "*" and self.resource != resource:
            # Check prefix match
            if not resource.startswith(self.resource.rstrip("*")):
                return False

        # Check action match
        if self.action != PermissionAction.ADMIN and self.action != action:
            return False

        # Check conditions
        if self.conditions and context:
            for key, expected in self.conditions.items():
                if context.get(key) != expected:
                    return False

        return True


class Role(BaseModel):
    """A role with permissions.

    Attributes:
        id: Role identifier.
        name: Role name.
        description: Role description.
        permissions: Granted permissions.
        parent_roles: Inherited roles.
        metadata: Additional data.
    """

    id: str = Field(default_factory=lambda: f"ROLE-{uuid4().hex[:8].upper()}")
    name: str
    description: str = ""
    permissions: list[Permission] = Field(default_factory=list)
    parent_roles: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def has_permission(
        self,
        resource: str,
        action: PermissionAction,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Check if role has a permission.

        Args:
            resource: Resource to access.
            action: Action to perform.
            context: Request context.

        Returns:
            True if permitted.
        """
        return any(p.matches(resource, action, context) for p in self.permissions)

    def add_permission(
        self,
        resource: str,
        action: PermissionAction,
        conditions: dict[str, Any] | None = None,
    ) -> None:
        """Add a permission to the role.

        Args:
            resource: Resource identifier.
            action: Permitted action.
            conditions: Permission conditions.
        """
        self.permissions.append(
            Permission(
                resource=resource,
                action=action,
                conditions=conditions or {},
            )
        )


class Principal(BaseModel):
    """A security principal (user or service).

    Attributes:
        id: Principal identifier.
        name: Principal name.
        principal_type: Type (user, service, system).
        roles: Assigned role IDs.
        direct_permissions: Direct permission grants.
        attributes: Principal attributes.
        created_at: Creation timestamp.
        last_active: Last activity timestamp.
    """

    id: str
    name: str
    principal_type: str = "user"
    roles: list[str] = Field(default_factory=list)
    direct_permissions: list[Permission] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime | None = None


class AccessDecision(BaseModel):
    """Result of an access decision.

    Attributes:
        allowed: Whether access is allowed.
        principal_id: Principal requesting access.
        resource: Requested resource.
        action: Requested action.
        matching_permission: Permission that allowed access.
        reason: Decision reason.
        decided_at: Decision timestamp.
    """

    allowed: bool
    principal_id: str
    resource: str
    action: PermissionAction
    matching_permission: Permission | None = None
    reason: str = ""
    decided_at: datetime = Field(default_factory=datetime.utcnow)


class AccessController:
    """Role-based access controller.

    Manages principals, roles, and access decisions.

    Example:
        ```python
        from ununseptium.security import AccessController, Role, Permission

        controller = AccessController()

        # Create role with permissions
        admin_role = Role(
            name="admin",
            permissions=[
                Permission(resource="*", action="admin")
            ]
        )
        controller.add_role(admin_role)

        # Add principal
        controller.add_principal(Principal(
            id="user-1",
            name="Admin User",
            roles=["admin"]
        ))

        # Check access
        decision = controller.check_access("user-1", "transactions", "read")
        if decision.allowed:
            print("Access granted")
        ```
    """

    def __init__(self) -> None:
        """Initialize the controller."""
        self._roles: dict[str, Role] = {}
        self._principals: dict[str, Principal] = {}

    def add_role(self, role: Role) -> None:
        """Add a role.

        Args:
            role: Role to add.
        """
        self._roles[role.id] = role

    def get_role(self, role_id: str) -> Role | None:
        """Get a role by ID.

        Args:
            role_id: Role identifier.

        Returns:
            Role if found.
        """
        return self._roles.get(role_id)

    def add_principal(self, principal: Principal) -> None:
        """Add a principal.

        Args:
            principal: Principal to add.
        """
        self._principals[principal.id] = principal

    def get_principal(self, principal_id: str) -> Principal | None:
        """Get a principal by ID.

        Args:
            principal_id: Principal identifier.

        Returns:
            Principal if found.
        """
        return self._principals.get(principal_id)

    def assign_role(self, principal_id: str, role_id: str) -> bool:
        """Assign a role to a principal.

        Args:
            principal_id: Principal identifier.
            role_id: Role identifier.

        Returns:
            True if successful.
        """
        principal = self._principals.get(principal_id)
        if not principal:
            return False
        if role_id not in self._roles:
            return False
        if role_id not in principal.roles:
            principal.roles.append(role_id)
        return True

    def check_access(
        self,
        principal_id: str,
        resource: str,
        action: PermissionAction | str,
        context: dict[str, Any] | None = None,
    ) -> AccessDecision:
        """Check if a principal can access a resource.

        Args:
            principal_id: Principal identifier.
            resource: Resource to access.
            action: Action to perform.
            context: Request context.

        Returns:
            AccessDecision with result.
        """
        if isinstance(action, str):
            action = PermissionAction(action)

        principal = self._principals.get(principal_id)
        if not principal:
            return AccessDecision(
                allowed=False,
                principal_id=principal_id,
                resource=resource,
                action=action,
                reason="Principal not found",
            )

        # Check direct permissions first
        for perm in principal.direct_permissions:
            if perm.matches(resource, action, context):
                return AccessDecision(
                    allowed=True,
                    principal_id=principal_id,
                    resource=resource,
                    action=action,
                    matching_permission=perm,
                    reason="Direct permission grant",
                )

        # Check role permissions
        for role_id in principal.roles:
            role = self._roles.get(role_id)
            if role and role.has_permission(resource, action, context):
                matching = next(
                    (p for p in role.permissions if p.matches(resource, action, context)),
                    None,
                )
                return AccessDecision(
                    allowed=True,
                    principal_id=principal_id,
                    resource=resource,
                    action=action,
                    matching_permission=matching,
                    reason=f"Role permission: {role.name}",
                )

        return AccessDecision(
            allowed=False,
            principal_id=principal_id,
            resource=resource,
            action=action,
            reason="No matching permission",
        )

    def get_effective_permissions(
        self,
        principal_id: str,
    ) -> list[Permission]:
        """Get all effective permissions for a principal.

        Args:
            principal_id: Principal identifier.

        Returns:
            List of all permissions.
        """
        principal = self._principals.get(principal_id)
        if not principal:
            return []

        permissions: list[Permission] = list(principal.direct_permissions)

        for role_id in principal.roles:
            role = self._roles.get(role_id)
            if role:
                permissions.extend(role.permissions)

        return permissions
