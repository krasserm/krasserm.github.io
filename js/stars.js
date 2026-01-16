(function() {
    const dataEl = document.getElementById('stars-data');
    if (!dataEl) return;

    const CACHE_KEY = 'github_stars';
    const CACHE_TTL = 24 * 60 * 60 * 1000; // 24 hours

    // Normalize build data to standard structure
    const BUILD_DATA = {
        timestamp: parseInt(dataEl.dataset.buildTime, 10) || 0,
        stars: JSON.parse(dataEl.dataset.stars || '{}')
    };

    function getCache() {
        try {
            const cached = localStorage.getItem(CACHE_KEY);
            if (!cached) return null;
            return JSON.parse(cached);
        } catch (e) {
            return null;
        }
    }

    function setCache(data) {
        try {
            localStorage.setItem(CACHE_KEY, JSON.stringify(data));
        } catch (e) {
            // localStorage unavailable (private mode, etc.)
        }
    }

    function age(data) {
        return Date.now() - data.timestamp;
    }

    function hasStars(data) {
        return data && data.stars && Object.keys(data.stars).length > 0;
    }

    function formatCount(count) {
        return count >= 1000 ? (count / 1000).toFixed(1) + 'k' : count;
    }

    function displayStars(data) {
        if (!hasStars(data)) return;
        const links = document.querySelectorAll('.project-stars-link[data-repo]');
        links.forEach(link => {
            const repo = link.dataset.repo;
            if (data.stars[repo] !== undefined) {
                link.querySelector('.star-count').textContent = formatCount(data.stars[repo]);
            }
        });
    }

    function fetchFreshStars(currentData) {
        const links = document.querySelectorAll('.project-stars-link[data-repo]');
        const repos = Array.from(links).map(l => l.dataset.repo);

        return Promise.all(repos.map(repo =>
            fetch(`https://api.github.com/repos/${repo}`)
                .then(r => r.ok ? r.json() : null)
                .then(data => {
                    if (data && data.stargazers_count !== undefined) {
                        return { repo, count: data.stargazers_count };
                    }
                    return null;
                })
                .catch(() => null)
        )).then(results => {
            const fetchedStars = {};
            results.forEach(r => {
                if (r) fetchedStars[r.repo] = r.count;
            });

            if (Object.keys(fetchedStars).length === 0) {
                return null;
            }

            // Merge: current stars + fetched stars (fetched overwrites)
            return {
                timestamp: Date.now(),
                stars: { ...(currentData?.stars || {}), ...fetchedStars }
            };
        });
    }

    // Main logic
    const bd = hasStars(BUILD_DATA) ? BUILD_DATA : null;
    const cd = getCache();

    // Determine best available data
    let d = bd;
    if (cd && (!d || d.timestamp < cd.timestamp)) {
        d = cd;
    }

    // Immediate display with best available data
    displayStars(d);

    // Fetch if no data or data is stale
    if (!d || age(d) > CACHE_TTL) {
        fetchFreshStars(d).then(fd => {
            if (fd) {
                d = fd;
                displayStars(d);
            }
            // Save best known state to cache
            if (d) setCache(d);
        });
    } else {
        // Save best known state to cache (handles bd newer than cd case)
        if (d) setCache(d);
    }
})();
