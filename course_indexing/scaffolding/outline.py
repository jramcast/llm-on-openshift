import yaml


def load_outline(path):
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.SafeLoader)
    return Outline(y)


class Outline:
    def __init__(self, data):
        self.chapters = [Chapter(c, i) for i, c in enumerate(data["dco"]["chapters"])]
        if "compreview" in data["dco"]:
            self.chapters.append(
                compreview_to_chapter(data["dco"]["compreview"], len(self.chapters))
            )
        bookinfo = data["dco"]["bookinfo"] if "bookinfo" in data["dco"] else data["dco"]
        self.sku = bookinfo["sku"]
        self.title = bookinfo["title"]
        self.product_name = bookinfo["product_name"]
        self.product_number = bookinfo["product_number"]
        self.edition = bookinfo["edition"]
        self.edition_date = bookinfo["edition_date"]
        self.authors = bookinfo["authors"]
        self.editors = bookinfo["editors"]
        self.contributors = bookinfo.get("contributors", [])
        self.architects = bookinfo["architects"]
        self.full_contributors = bookinfo.get("devops", []) + self.contributors
        self.scaffolding_version = data["skeleton"]["revision"]

    def find_section_id_by_title(self, chapter_title: str, section_title: str):
        chapter_title = chapter_title.lower()
        section_title = (
            section_title.replace("Guided Exercise ", "")
            .replace("Lab ", "")
            .replace("Matching ", "")
            .replace("Quiz ", "")
            .strip()
            .lower()
        )

        if chapter_title == "introduction":
            return "pr01"

        if chapter_title == "":
            return None

        if chapter_title == "comprehensive review":
            return f"ch{len(self.chapters)}"

        chapter_titles = [c.title.lower() for c in self.chapters]


        chapter_index = chapter_titles.index(chapter_title)
        chapter = self.chapters[chapter_index]

        if section_title == "":
            return f"ch{chapter.number:02}"

        if section_title == "summary":
            summary_section_number = len(chapter.sections) + 1
            return f"ch{chapter.number:02}s{summary_section_number:02}"

        section_titles = [s.topic.title.lower() for s in chapter.sections]
        print(section_titles)
        section_index = section_titles.index(section_title)
        section = chapter.sections[section_index]
        return section.id


def compreview_to_chapter(compreview, index):
    # HACK! compreviews are not chapters, but they are very similar

    # these fields are not present in compreviews, but we add them for uniformity
    compreview["keyword"] = "compreview"
    compreview["title"] = "Comprehensive Review"
    # those are optional in compreviews (but mandatory in chapters)
    compreview["goal"] = None
    compreview["objectives"] = None
    return Chapter(compreview, index)


class Chapter:
    def __init__(self, data, index):
        self.keyword = data["keyword"]
        self.title = data.get("title")
        self.index = index
        self.number = index + 1
        self.goal = data.get("goal")
        self.objectives = data.get("objectives")
        self.topics = [Topic(t, self) for t in data.get("topics", [])]

        for i, section in enumerate(self.sections):
            section.number = i + 1

    def __str__(self):
        return f"ch{self.number:02}-{self.keyword}"

    @property
    def sections(self):
        return [s for t in self.topics for s in t.sections]


class Topic:
    def __init__(self, data, chapter: Chapter):
        self.keyword = data["keyword"]
        self.title = data.get("title")
        self.chapter = chapter
        self.sections = [Section(s, self) for s in data.get("sections", [])]

    def __str__(self):
        return f"ch{self.chapter.number:02}-{self.chapter.keyword}-{self.keyword}"


class Section:
    number: int

    def __init__(self, data, topic: Topic):
        self.type = data["type"]
        self.status = data.get("status")
        self.time = data.get("time")
        self.topic = topic
        self.objectives = data.get("objectives")
        self.path = f"content/{self.topic.chapter.keyword}/{self.topic.keyword}/{self.type}.adoc"
        self.scope = data.get("scope", {})

    @property
    def course(self):
        return self.scope.get("course", True)

    @property
    def id(self):
        return f"ch{self.topic.chapter.number:02}s{self.number:02}"

    def __str__(self):
        return (
            f"{self.id}-"
            f"{self.topic.chapter.keyword}-{self.topic.keyword}-{self.type}"
        )


LAB_TYPES = (
    "practice",
    "lab",
    "review",
    "ge",
)
GRADED_LAB_TYPES = (
    "lab",
    "review",
)


def outline_to_csv():
    outline = load_outline("outline.yml")
    print(
        "\t".join(
            [
                "chapter number",
                "chapter keyword",
                "chapter title",
                "topic keyword",
                "topic title",
                "section",
                "section type",
                "time",
                "path",
                "status",
            ]
        )
    )
    for ci, c in enumerate(outline.chapters):
        si = 0
        for t in c.topics:
            for s in t.sections:
                si += 1
                print(
                    "\t".join(
                        map(
                            str,
                            [
                                ci + 1,
                                c.keyword,
                                c.title,
                                t.keyword,
                                t.title,
                                si,
                                s.type,
                                s.time,
                                s.path,
                                s.status,
                            ],
                        )
                    )
                )
