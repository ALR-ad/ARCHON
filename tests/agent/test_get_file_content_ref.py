import pytest
from unittest.mock import patch, MagicMock
from agent.github_client import get_file_content

@patch('agent.github_client._get_github_client')
def test_get_file_content_pins_commit(mock_get_client):
    mock_gh = MagicMock()
    mock_get_client.return_value = mock_gh
    mock_repo = MagicMock()
    mock_gh.get_repo.return_value = mock_repo
    
    def mock_get_contents(file_path, ref):
        mock_content = MagicMock()
        if ref == 'sha_old':
            mock_content.decoded_content = b'old content'
        elif ref == 'sha_new':
            mock_content.decoded_content = b'new content'
        else:
            mock_content.decoded_content = b'default content'
        return mock_content
        
    mock_repo.get_contents.side_effect = mock_get_contents
    
    content_old = get_file_content('test/repo', 'test.py', 'sha_old')
    content_new = get_file_content('test/repo', 'test.py', 'sha_new')
    
    assert content_old == 'old content'
    assert content_new == 'new content'
    
    mock_repo.get_contents.assert_any_call('test.py', ref='sha_old')
    mock_repo.get_contents.assert_any_call('test.py', ref='sha_new')
    print('TEST PASSED: get_file_content successfully pins commit SHAs!')
    
if __name__ == '__main__':
    pytest.main(['-v', __file__])
