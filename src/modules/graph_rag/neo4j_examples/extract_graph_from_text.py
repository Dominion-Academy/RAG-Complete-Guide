from datetime import date, datetime
import os
from string import Template

from neo4j import GraphDatabase, NotificationMinimumSeverity
from pydantic import BaseModel, Field

from src.components.llm import InputMessage, SyncOpenAILikeLLM
from src.components.utils import repair_json_string
from src.settings import DATASETS_DIR, settings


class BaseEntity(BaseModel):
    slug: str = Field(description="Идентификатор сущности")
    name: str = Field(description="Каноническое имя/название сущности")
    aliases: list[str] = Field(default_factory=list, description="Альтернативные названия/синонимы")
    description: str | None = Field(default=None, description="Краткое описание сущности на основе текста")


class Person(BaseEntity):
    """Конкретный человек"""

    birth_date: date | None = Field(default=None, description="Дата рождения, если указана")
    death_date: date | None = Field(default=None, description="Дата смерти, если указана")
    occupation: str | None = Field(default=None, description="Род занятий/профессия")


class Organization(BaseEntity):
    """Компании, учреждения и другие организации."""

    founded_date: date | None = Field(default=None, description="Дата основания")
    industry: str | None = Field(default=None, description="Сфера работы")
    members: list[str] = Field(default_factory=list, description="Slug связанных Person-сотрудников")


class Location(BaseEntity):
    """Географические места."""

    country: str | None = Field(default=None, description="Страна, если известно")
    region: str | None = Field(default=None, description="Регион/штадт страны, если известно")
    city: str | None = Field(default=None, description="Город страны, если известно")
    address: str | None = Field(default=None, description="Улица, дом, квартира, если известно")
    part_of: str | None = Field(default=None, description="Slug Location, если известно")


class Event(BaseEntity):
    """События, произошедшие (или происходящие) в тексте."""

    start_date: date | datetime | None = Field(default=None, description="Дата и время старта события")
    end_date: date | datetime | None = Field(default=None, description="Дата и время окончания события")
    location: str | None = Field(default=None, description="Slug Location, где произошло")
    participants: list[str] = Field(
        default_factory=list,
        description="Slug Person/Organization, участвовавших в событии",
    )
    objects_involved: list[str] = Field(default_factory=list, description="Slug Object, задействованных в событии")
    causes: list[str] = Field(default_factory=list, description="Причины события")
    consequences: list[str] = Field(default_factory=list, description="Последствия события")


class Object(BaseEntity):
    """Физические или цифровые объекты, а также всё остальное."""

    owner: str | None = Field(default=None, description="Slug Person-владельца события")
    location: str | None = Field(default=None, description="Slug Location")


class ExtractionResult(BaseModel):
    """Результат извлечения всех сущностей из текста."""

    persons: list[Person] = Field(default_factory=list, description="Все люди")
    organizations: list[Organization] = Field(default_factory=list, description="Все организации")
    locations: list[Location] = Field(default_factory=list, description="Все места")
    events: list[Event] = Field(default_factory=list, description="Все события")
    objects: list[Object] = Field(default_factory=list, description="Все объекты")


SYSTEM_PROMPT = Template(
    """
    You are an expert in extracting structured information from texts. 
    Present the extracted information in a clear, structured format. 
    Follow the strict schema: $schema
    """
)


if __name__ == "__main__":
    raw_text = ""
    DATASET_DIR = DATASETS_DIR / "ai_helpdesk" / "data"
    files = os.listdir(DATASET_DIR)
    for file in files:
        file_path = os.path.join(DATASET_DIR, file)
        with open(file_path, encoding="utf-8") as f:
            file_text = f.read()
        raw_text += " " + file_text

    llm = SyncOpenAILikeLLM(
        base_url=settings.llm.BASE_URL,
        api_key=settings.llm.API_KEY,
        model=settings.llm.MODEL,
        common_parameters={"temperature": 0},
    )

    messages = [
        InputMessage(role="system", content=SYSTEM_PROMPT.substitute(schema=ExtractionResult.model_json_schema())),
        InputMessage(role="user", content=raw_text),
    ]
    answer = llm.generate_answer(messages)
    answer_content = answer.content
    print(answer_content)
    repaired_json_string = repair_json_string(answer_content)
    result = ExtractionResult.model_validate_json(repaired_json_string)

    # Neo4j
    driver = GraphDatabase.driver(
        "neo4j://localhost:7687",
        auth=("neo4j", "neo4j-password"),
        notifications_min_severity=NotificationMinimumSeverity.OFF,
    )

    # Entities
    rel_fields = {"members", "part_of", "location", "participants", "objects_involved", "owner"}
    for entity in [*result.persons, *result.locations, *result.objects, *result.events, *result.organizations]:
        entity_dict = entity.model_dump()
        for field in rel_fields:
            if field in entity_dict:
                del entity_dict[field]
        driver.execute_query(
            f"MERGE (n:Entity:{entity.__class__.__name__} {{slug: $row.slug}})SET n += $row",
            row=entity_dict,
        )

    # Organization.members
    for organization in result.organizations:
        for member_slug in organization.members:
            driver.execute_query(
                """
                MATCH (o:Organization {slug: $org_slug})
                MATCH (p:Person {slug: $member_slug})
                MERGE (p)-[:IS_MEMBER_OF]->(o)
                """,
                org_slug=organization.slug,
                member_slug=member_slug,
            )

    # Location.part_of
    for location in result.locations:
        if location.part_of:
            driver.execute_query(
                """
                MATCH (child:Location {slug: $child_slug})
                MATCH (parent:Location {slug: $parent_slug})
                MERGE (child)-[:IS_PART_OF]->(parent)
                """,
                child_slug=location.slug,
                parent_slug=location.part_of,
            )

    # Event.location
    for event in result.events:
        if event.location:
            driver.execute_query(
                """
                MATCH (e:Event {slug: $event_slug})
                MATCH (l:Location {slug: $location_slug})
                MERGE (e)-[:LOCATED_AT]->(l)
                """,
                event_slug=event.slug,
                location_slug=event.location,
            )

    # Event.participants: Person или Organization
    for event in result.events:
        for participant_slug in event.participants:
            driver.execute_query(
                """
                MATCH (e:Event {slug: $event_slug})
                MATCH (p:Entity {slug: $participant_slug})
                WHERE p:Person OR p:Organization
                MERGE (p)-[:PARTICIPATED_IN]->(e)
                """,
                event_slug=event.slug,
                participant_slug=participant_slug,
            )

    # Event.objects_involved
    for event in result.events:
        for object_slug in event.objects_involved:
            driver.execute_query(
                """
                MATCH (e:Event {slug: $event_slug})
                MATCH (o:Object {slug: $object_slug})
                MERGE (o)-[:INVOLVED_IN]->(e)
                """,
                event_slug=event.slug,
                object_slug=object_slug,
            )

    # Object.owner
    for obj in result.objects:
        if obj.owner:
            driver.execute_query(
                """
                MATCH (o:Object {slug: $object_slug})
                MATCH (p:Person {slug: $owner_slug})
                MERGE (o)-[:OWNED_BY]->(p)
                """,
                object_slug=obj.slug,
                owner_slug=obj.owner,
            )

    # Object.location
    for obj in result.objects:
        if obj.location:
            driver.execute_query(
                """
                MATCH (o:Object {slug: $object_slug})
                MATCH (l:Location {slug: $location_slug})
                MERGE (o)-[:LOCATED_AT]->(l)
                """,
                object_slug=obj.slug,
                location_slug=obj.location,
            )
