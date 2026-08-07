"""Seed synthetic approved evidence into PostgreSQL."""

from careerops_agent_engine.infrastructure.database.models.evidence import (
    CareerEvidenceRecord,
)
from careerops_agent_engine.infrastructure.database.session import (
    create_database_engine,
    create_session_factory,
)
from careerops_agent_engine.infrastructure.repositories.development_evidence import (
    DEVELOPMENT_USER_ID,
    create_development_evidence_repository,
)


def main() -> None:
    """Insert or update the synthetic development evidence."""

    engine = create_database_engine()
    session_factory = create_session_factory(engine)

    development_repository = create_development_evidence_repository()
    evidence_records = development_repository.list_approved(
        user_id=DEVELOPMENT_USER_ID,
    )

    try:
        with session_factory() as session:
            for evidence in evidence_records:
                session.merge(
                    CareerEvidenceRecord(
                        user_id=DEVELOPMENT_USER_ID,
                        evidence_id=evidence.evidence_id,
                        category=evidence.category.value,
                        title=evidence.title,
                        verification_status=(evidence.verification_status.value),
                        technologies=list(evidence.technologies),
                        capabilities=list(evidence.capabilities),
                        approved_claims=list(evidence.approved_claims),
                        source_references=[
                            {
                                "source_type": (reference.source_type.value),
                                "source_id": reference.source_id,
                                "page_number": (reference.page_number),
                                "source_excerpt": (reference.source_excerpt),
                            }
                            for reference in evidence.source_references
                        ],
                    )
                )

            session.commit()

        print(
            "Seeded "
            f"{len(evidence_records)} development evidence records "
            f"for {DEVELOPMENT_USER_ID}."
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
