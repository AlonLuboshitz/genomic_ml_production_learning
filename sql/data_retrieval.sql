-- ============================================================
-- SQL data retrieval — queries to extract training datasets
--
-- Use the genomics_practice.db database.
-- Tables: variants (chrom, pos, ref, alt, sample_id, genotype)
--         annotations (chrom, pos, ref, alt, gene, impact, consequence)
--
-- Open the database:
--   sqlite3 teaching_examples/genomics_practice.db
-- ============================================================

-- ────────────────────────────────────────────────────────────
-- Exercise 1: Basic filtering
-- Write a SELECT that returns all variants on chromosome 1
-- where the sample is 'S002'.
-- Expected: 2 rows
-- ────────────────────────────────────────────────────────────
SELECT * FROM variants WHERE chrom = 'chr1' AND sample_id = 'S002';

-- ────────────────────────────────────────────────────────────
-- Exercise 2: JOIN with WHERE filter
-- Return sample_id, gene, impact, and genotype for all
-- variants with impact = 'HIGH' (only nonsense variants).
-- Expected: 3 rows (one per sample for the BRCA1 nonsense)
-- ────────────────────────────────────────────────────────────
SELECT
    v.sample_id,
    v.genotype,
    a.gene,
    a.impact
FROM variants v
JOIN annotations a ON a.chrom = v.chrom
                  AND a.pos   = v.pos
                  AND a.ref   = v.ref
                  AND a.alt   = v.alt
WHERE a.impact = 'HIGH';

-- ────────────────────────────────────────────────────────────
-- Exercise 3: Aggregation — per-gene stats
-- For each gene, return:
--   gene, num_samples_with_variant, avg_genotype
-- Hint: COUNT(DISTINCT sample_id) for unique samples.
-- Expected: 2 rows (GENE_A, BRCA1)
-- ────────────────────────────────────────────────────────────
SELECT
    a.gene,
    COUNT(DISTINCT v.sample_id)   AS num_samples_with_variant,
    AVG(v.genotype)               AS avg_genotype
FROM variants v
JOIN annotations a ON a.chrom = v.chrom
                  AND a.pos   = v.pos
                  AND a.ref   = v.ref
                  AND a.alt   = v.alt
GROUP BY a.gene;

-- ────────────────────────────────────────────────────────────
-- Exercise 4: ML feature matrix — one row per sample
-- Return sample_id plus one column per gene showing the
-- genotype value (use MAX(CASE ...) pivot).
-- Expected: 3 rows (S001, S002, S003)
-- ────────────────────────────────────────────────────────────
SELECT
    v.sample_id,
    MAX(CASE WHEN a.gene = 'GENE_A' THEN v.genotype END) AS GENE_A_genotype,
    MAX(CASE WHEN a.gene = 'BRCA1'  THEN v.genotype END) AS BRCA1_genotype
FROM variants v
JOIN annotations a ON a.chrom = v.chrom
                  AND a.pos   = v.pos
                  AND a.ref   = v.ref
                  AND a.alt   = v.alt
GROUP BY v.sample_id
ORDER BY v.sample_id;

-- ────────────────────────────────────────────────────────────
-- Exercise 5 (Bonus): Count HIGH-impact variants per sample
-- Return sample_id and high_impact_count for all samples.
-- Expected: S001=1, S002=1, S003=1
-- ────────────────────────────────────────────────────────────
SELECT
    v.sample_id,
    COUNT(*) AS high_impact_count
FROM variants v
JOIN annotations a ON a.chrom = v.chrom
                  AND a.pos   = v.pos
                  AND a.ref   = v.ref
                  AND a.alt   = v.alt
WHERE a.impact = 'HIGH'
GROUP BY v.sample_id
ORDER BY v.sample_id;
