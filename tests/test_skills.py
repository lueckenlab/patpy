from importlib.resources import files


def test_skill_markdown_files_are_packaged():
    expected_files = [
        "SKILL.md",
        "cellxgene/SKILL.md",
        "datasets/SKILL.md",
        "evaluation/SKILL.md",
        "plotting/SKILL.md",
        "preprocessing/SKILL.md",
        "sample_representation/SKILL.md",
        "supervised_methods/SKILL.md",
    ]

    skill_root = files("patpy.skills")
    for relative_path in expected_files:
        resource = skill_root
        for segment in relative_path.split("/"):
            resource = resource.joinpath(segment)

        assert resource.is_file()
        assert resource.read_text(encoding="utf-8").startswith("---")
