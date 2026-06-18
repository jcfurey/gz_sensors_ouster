"""vcstool workspace file sanity checks.

Verifies gz_sensors_ouster.repos is valid YAML and contains the expected
repository entries with all required fields.  Catches accidentally malformed
edits to the repos file that would break workspace setup for users.
"""
import pathlib

import pytest
import yaml

REPOS_FILE = pathlib.Path(__file__).resolve().parent.parent / 'gz_sensors_ouster.repos'


@pytest.fixture(scope='module')
def repos():
    return yaml.safe_load(REPOS_FILE.read_text())


def test_repos_file_exists():
    assert REPOS_FILE.exists(), f'{REPOS_FILE} does not exist'


def test_repos_is_valid_yaml(repos):
    assert isinstance(repos, dict)


def test_repos_has_repositories_key(repos):
    assert 'repositories' in repos


def test_gz_sensors_ouster_entry_exists(repos):
    assert 'gz_sensors_ouster' in repos['repositories']


def test_gz_sensors_ouster_fields(repos):
    r = repos['repositories']['gz_sensors_ouster']
    assert r.get('type') == 'git'
    assert 'github.com/jcfurey/gz_sensors_ouster' in r.get('url', '')
    assert isinstance(r.get('version'), str) and r['version']


def test_ouster_ros_entry_exists(repos):
    assert 'ouster-ros' in repos['repositories']


def test_ouster_ros_fields(repos):
    r = repos['repositories']['ouster-ros']
    assert r.get('type') == 'git'
    assert 'github.com/jcfurey/ouster-ros' in r.get('url', '')
    version = r.get('version', '')
    assert isinstance(version, str) and len(version) >= 8, (
        f'version should be a pinned SHA or branch, got: {version!r}')


def test_all_entries_have_required_fields(repos):
    for name, entry in repos['repositories'].items():
        assert 'type' in entry, f'{name}: missing type'
        assert 'url' in entry, f'{name}: missing url'
        assert 'version' in entry, f'{name}: missing version'
