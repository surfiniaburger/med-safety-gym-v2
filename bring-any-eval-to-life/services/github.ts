
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

interface GitHubContent {
    name: string;
    path: string;
    sha: string;
    type: string;
    url: string;
    html_url: string;
    download_url: string;
}

export async function fetchEvaluationArtifacts(): Promise<EvaluationArtifact[]> {
    try {
        // 1. Fetch list of files in the results directory
        const response = await fetch(
            `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents/${RESULTS_PATH}`
        );

        if (!response.ok) {
            if (response.status === 404) {
                console.warn('Results directory not found, returning empty list.');
                return [];
            }
            throw new Error(`GitHub API error: ${response.statusText}`);
        }

        const files = await response.json();

        if (!Array.isArray(files)) {
            console.warn('GitHub API returned non-array for contents');
            return [];
        }

        // 2. Filter for .json files and sort by name descending (latest first)
        const jsonFiles = files
            .filter((file: any): file is GitHubContent =>
                typeof file.name === 'string' &&
                file.name.endsWith('.json') &&
                file.type === 'file'
            )
            .sort((a, b) => b.name.localeCompare(a.name)) // Latest first
            .slice(0, 5); // Only latest 5

        // 3. Fetch content for each file
        const artifacts: EvaluationArtifact[] = await Promise.all(
            jsonFiles.map(async (file) => {
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
                    const contentResponse = await fetch(file.download_url);
                    if (contentResponse.ok) {
                        const jsonContent = await contentResponse.json();
                        artifact.content = jsonContent as EvaluationContent;
                    } else {
                        console.error(`Failed to fetch content for ${file.name}: ${contentResponse.statusText}`);
                    }
                } catch (err) {
                    console.error(`Error parsing content for ${file.name}`, err);
                }

                return artifact;
            })
        );

        // Sort by name or date if needed (optional)
        return artifacts;

    } catch (error) {
        console.error('Failed to fetch evaluation artifacts:', error);
        // Return empty array or throw based on preference. 
        // For UI resilience, returning empty array with logged error is often better, 
        // but throwing allows React Query to handle error state.
        // Let's throw to let the caller handle the error state.
        throw error;
    }
}
