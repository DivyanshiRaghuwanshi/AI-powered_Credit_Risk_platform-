from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CustomerEntity:
    customer_gid: str
    source_customer_id: str
    age_years: Optional[float] = None
    income_total: Optional[float] = None
    region_gid: Optional[str] = None
    segment: Optional[str] = None


@dataclass(frozen=True)
class LoanEntity:
    loan_gid: str
    customer_gid: str
    principal_amount: Optional[float] = None
    annuity_amount: Optional[float] = None
    product_type: Optional[str] = None
    state: Optional[str] = None


@dataclass(frozen=True)
class ApplicationEntity:
    application_gid: str
    customer_gid: str
    status: Optional[str] = None
    requested_amount: Optional[float] = None
    decision_ts: Optional[str] = None


@dataclass(frozen=True)
class BureauRecordEntity:
    bureau_record_gid: str
    customer_gid: str
    overdue_days: Optional[float] = None
    credit_sum_debt: Optional[float] = None


@dataclass(frozen=True)
class AnalystDocumentEntity:
    doc_gid: str
    customer_gid: Optional[str] = None
    doc_type: Optional[str] = None
    content: Optional[str] = None
