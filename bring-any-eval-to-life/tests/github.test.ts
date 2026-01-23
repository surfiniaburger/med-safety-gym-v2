import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fetchEvaluationArtifacts } from '../services/github';

describe('GitHub Fetcher Service', () => {
    beforeEach(() => {
        vi.resetAllMocks();
    });

    it('fetches and processes evaluation artifacts correctly', async () => {
        // 1. Mock the first fetch (list of files)
        const mockFilesResponse = [
            {
                name: 'result-1.json',
                sha: 'sha1',
                type: 'file',
                url: 'https://api.github.com/content/1',
                html_url: 'https://github.com/1',
                download_url: 'https://raw.githubusercontent.com/1',
            },
            {
                name: 'not-a-json.txt',
                sha: 'sha2',
                type: 'file',
                url: 'https://api.github.com/content/2',
                html_url: 'https://github.com/2',
                download_url: 'https://raw.githubusercontent.com/2',
            }
        ];

        // 2. Mock the second fetch (content of the JSON file)
        const mockContentResponse = {
            safety_score: 0.95,
            status: 'SAFE',
            summary: 'Test summary'
        };

        const globalFetch = vi.fn();
        global.fetch = globalFetch;

        globalFetch
            .mockResolvedValueOnce({
                ok: true,
                json: async () => mockFilesResponse,
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => mockContentResponse,
            });

        const artifacts = await fetchEvaluationArtifacts();

        expect(artifacts).toHaveLength(1);
        expect(artifacts[0].name).toBe('result-1.json');
        expect(artifacts[0].content?.safety_score).toBe(0.95);
        expect(artifacts[0].content?.status).toBe('SAFE');
    });

    it('handles empty results directory', async () => {
        const globalFetch = vi.fn();
        global.fetch = globalFetch;

        globalFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => [],
        });

        const artifacts = await fetchEvaluationArtifacts();
        expect(artifacts).toHaveLength(0);
    });

    it('handles 404 error from GitHub API', async () => {
        const globalFetch = vi.fn();
        global.fetch = globalFetch;

        globalFetch.mockResolvedValueOnce({
            ok: false,
            status: 404,
            statusText: 'Not Found',
        });

        const artifacts = await fetchEvaluationArtifacts();
        expect(artifacts).toHaveLength(0);
    });
});
