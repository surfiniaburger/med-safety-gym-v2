
export interface EvaluationArtifact {
    id: string;
    name: string;
    path: string;
    sha: string;
    url: string; // GitHub API URL
    html_url: string; // Browser URL
    download_url: string; // Raw content URL
    content?: EvaluationContent;
}

export interface EvaluationContent {
    safety_score: number;
    status: 'SAFE' | 'UNSAFE' | 'UNKNOWN';
    summary: string;
    // Add other fields as they appear in the JSON
    [key: string]: any;
}

const GITHUB_OWNER = 'surfiniaburger';
const GITHUB_REPO = 'med-safety-gym';
const RESULTS_PATH = 'results';
const CACHE_KEY = 'github_artifacts_cache';

interface GitHubContent {
    name: string;
    path: string;
    sha: string;
    type: string;
    url: string;
    html_url: string;
    download_url: string;
}

interface CacheData {
    data: EvaluationArtifact[];
    etag: string | null;
    timestamp: number;
}

export async function fetchEvaluationArtifacts(): Promise<EvaluationArtifact[]> {
    try {
        // 1. Check cache
        const cached = localStorage.getItem(CACHE_KEY);
        let cacheData: CacheData | null = null;
        if (cached) {
            try {
                cacheData = JSON.parse(cached);
            } catch (e) {
                console.error('Failed to parse cache', e);
            }
        }

        const headers: Record<string, string> = {};
        if (cacheData?.etag) {
            headers['If-None-Match'] = cacheData.etag;
        }

        // 2. Fetch list of files in the results directory
        const response = await fetch(
            `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents/${RESULTS_PATH}`,
            { headers }
        );

        // 3. Handle 304 Not Modified
        if (response.status === 304 && cacheData) {
            return cacheData.data;
        }

        if (!response.ok) {
            if (response.status === 404) {
                console.warn('Results directory not found, returning empty list.');
                return [];
            }
            throw new Error(`GitHub API error: ${response.statusText}`);
        }

        const etag = response.headers.get('ETag');
        const files = await response.json();

        if (!Array.isArray(files)) {
            console.warn('GitHub API returned non-array for contents');
            return [];
        }

        // 4. Filter for .json files and sort by name descending (latest first)
        const jsonFiles = files
            .filter((file: any): file is GitHubContent =>
                typeof file.name === 'string' &&
                file.name.endsWith('.json') &&
                file.type === 'file'
            )
            .sort((a, b) => b.name.localeCompare(a.name)) // Latest first
            .slice(0, 5); // Only latest 5

        // 5. Fetch content for each file
        const artifacts: EvaluationArtifact[] = await Promise.all(
            jsonFiles.map(async (file) => {
                // Check if we already have this file in cache by SHA
                const cachedArtifact = cacheData?.data.find(a => a.sha === file.sha);
                if (cachedArtifact && cachedArtifact.content) {
                    return cachedArtifact;
                }

                const artifact: EvaluationArtifact = {
                    id: file.sha,
                    name: file.name,
                    path: file.path,
                    sha: file.sha,
                    url: file.url,
                    html_url: file.html_url,
                    download_url: file.download_url,
                };

                try {
                    // Use the GitHub API URL instead of raw.githubusercontent.com to avoid timeouts
                    const contentResponse = await fetch(file.url);
                    if (contentResponse.ok) {
                        const fileData = await contentResponse.json();
                        if (fileData.content) {
                            // Decode base64 content
                            const base64Content = fileData.content.replace(/\s/g, '');
                            const decodedString = atob(base64Content);
                            artifact.content = JSON.parse(decodedString) as EvaluationContent;
                        } else {
                            console.error(`Received no content from API for ${file.name}`);
                        }
                    } else {
                        console.error(`Failed to fetch content via API for ${file.name}: ${contentResponse.statusText}`);
                    }
                } catch (err) {
                    console.error(`Error processing content via API for ${file.name}`, err);
                }

                return artifact;
            })
        );

        // 6. Update cache
        const newCacheData: CacheData = {
            data: artifacts,
            etag: etag,
            timestamp: Date.now()
        };
        localStorage.setItem(CACHE_KEY, JSON.stringify(newCacheData));

        return artifacts;

    } catch (error) {
        console.error('Failed to fetch evaluation artifacts:', error);
        throw error;
    }
}
