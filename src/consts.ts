// Place any global data in this file.
// You can import this data from anywhere in your site by using the `import` keyword.

// Base Page Metadata, src/layouts/BaseLayout.astro
export const BRAND_NAME = "UCCL";
export const SITE_TITLE = "UCCL";
export const SITE_DESCRIPTION = "An amazing team!";

// Tags Page Metadata, src/pages/tags/index.astro
export const Tags_TITLE = "UCCL - All Tags";
export const Tags_DESCRIPTION =
  "UCCL - All tags and the count of articles related to each tag";

// Tags Page Metadata, src/pages/tags/[tag]/[page].astro
export function getTagMetadata(tag: string) {
  return {
    title: `All articles on '${tag}' tag in UCCL`,
    description: `Explore articles about ${tag} for different perspectives and in-depth analysis.`,
  };
}

// Category Page Metadata, src/pages/category/[category]/[page].astro
export function getCategoryMetadata(category: string) {
  return {
    title: `All articles in '${category}' category in UCCL`,
    description: `Browse all articles under the ${category} category in UCCL`,
  };
}

// Header Links, src/components/Header.astro
export const HeaderLinks = [
  { href: "/category/One/1/", title: "Blog Posts" },
];

// Footer Links, src/components/Footer.astro
export const FooterLinks = [
  { href: "https://sky.cs.berkeley.edu/", title: "Sky Computing Lab @ Berkeley" },
  { href: "/tags/", title: "Tags" },
];

// Social Links, src/components/Footer.astro
export const SocialLinks = [
  {
    href: "https://github.com/uccl-project",
    icon: "tabler:brand-github",
    label: "GitHub",
  },
];

// Search Page Metadata, src/pages/search.astro
export const SEARCH_PAGE_TITLE = `${SITE_TITLE} - Site Search`;
export const SEARCH_PAGE_DESCRIPTION = `Search all content on ${SITE_TITLE}`;
