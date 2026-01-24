import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fetchEvaluationArtifacts } from '../services/github';

describe('GitHub Fetcher Service', () => {
    beforeEach(() => {
        vi.resetAllMocks();
        localStorage.clear();
    });

    afterEach(() => {
        vi.restoreAllMocks();
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

        // 2. Mock the second fetch (content of the JSON file via GitHub API)
        const mockContentBody = {
            safety_score: 0.95,
            status: 'SAFE',
            summary: 'Test summary'
        };
        const mockApiResponse = {
            content: btoa(JSON.stringify(mockContentBody))
        };

        const globalFetch = vi.fn();
        global.fetch = globalFetch;

        globalFetch
            .mockResolvedValueOnce({
                ok: true,
                headers: new Headers({ 'ETag': 'test-etag' }),
                json: async () => mockFilesResponse,
            })
            .mockResolvedValueOnce({
                ok: true,
                headers: new Headers(),
                json: async () => mockApiResponse,
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
            headers: new Headers(),
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

    describe('Caching', () => {
        it('should return cached data if available and not modified (304)', async () => {
            const mockCachedData = [
                {
                    id: 'cached-sha',
                    name: 'cached-result.json',
                    path: 'results/cached-result.json',
                    sha: 'cached-sha',
                    url: 'https://api.github.com/content/cached',
                    html_url: 'https://github.com/cached',
                    download_url: 'https://raw.githubusercontent.com/cached',
                    content: { safety_score: 1.0, status: 'SAFE', summary: 'Cached' }
                }
            ];

            // Set up localStorage with "cached" data
            localStorage.setItem('github_artifacts_cache', JSON.stringify({
                data: mockCachedData,
                timestamp: Date.now(),
                etag: 'some-etag'
            }));

            const globalFetch = vi.fn();
            global.fetch = globalFetch;

            // Mock a 304 Not Modified response
            globalFetch.mockResolvedValueOnce({
                ok: false,
                status: 304,
            });

            const artifacts = await fetchEvaluationArtifacts();

            // Should return cached data
            expect(artifacts).toEqual(mockCachedData);
            // Should have called fetch with the ETag
            expect(globalFetch).toHaveBeenCalledWith(
                expect.stringContaining('/contents/results'),
                expect.objectContaining({
                    headers: expect.objectContaining({
                        'If-None-Match': 'some-etag'
                    })
                })
            );
        });

        it('should fetch and cache data if cache is empty', async () => {
            const mockFilesResponse = [
                {
                    name: 'new-result.json',
                    sha: 'new-sha',
                    type: 'file',
                    url: 'https://api.github.com/content/new',
                    html_url: 'https://github.com/new',
                    download_url: 'https://raw.githubusercontent.com/new',
                }
            ];

            const mockApiResponse = {
                content: btoa(JSON.stringify({ safety_score: 0.8, status: 'SAFE', summary: 'New' }))
            };

            const globalFetch = vi.fn();
            global.fetch = globalFetch;

            globalFetch
                .mockResolvedValueOnce({
                    ok: true,
                    status: 200,
                    headers: new Headers({ 'ETag': 'new-etag' }),
                    json: async () => mockFilesResponse,
                })
                .mockResolvedValueOnce({
                    ok: true,
                    headers: new Headers(),
                    json: async () => mockApiResponse,
                });

            const artifacts = await fetchEvaluationArtifacts();

            expect(artifacts).toHaveLength(1);
            expect(artifacts[0].name).toBe('new-result.json');

            // Check if it was cached
            const cache = JSON.parse(localStorage.getItem('github_artifacts_cache') || '{}');
            expect(cache.etag).toBe('new-etag');
            expect(cache.data).toHaveLength(1);
            expect(cache.data[0].name).toBe('new-result.json');
        });
    });
});