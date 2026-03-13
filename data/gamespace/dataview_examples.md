# Dataview Query Examples

*Note: Requires the Dataview plugin to be installed*

## View All Parameters by Status

```dataview
TABLE survival_rating, status, tags
FROM "Parameters"
SORT status ASC
```

## High Overlap Parameters

```dataview
LIST
FROM "Parameters"
WHERE contains(tags, "high-overlap")
```

## Parameters to Keep

```dataview
TABLE survival_rating, tags
FROM "Parameters"
WHERE status = "keep"
```

## Undecided Parameters

```dataview
LIST
FROM "Parameters"
WHERE survival_rating = "undecided"
SORT file.name ASC
```

## Collapse Candidates

```dataview
TABLE related_parameters
FROM "Parameters"
WHERE collapse_candidate = true
```

## Parameters by Survival Rating

### High Survival Rating
```dataview
LIST
FROM "Parameters"
WHERE survival_rating = "high"
```

### Medium Survival Rating
```dataview
LIST
FROM "Parameters"
WHERE survival_rating = "medium"
```

### Low Survival Rating
```dataview
LIST
FROM "Parameters"
WHERE survival_rating = "low"
```

## Count by Status

```dataview
TABLE rows.file.link as "Parameters"
FROM "Parameters"
GROUP BY status
```

## All Gameplay Parameters

```dataview
LIST
FROM "Parameters"
WHERE contains(tags, "gameplay")
SORT file.name ASC
```

---

## How to Use Dataview

1. Install the Dataview plugin from Community Plugins
2. Copy any query above
3. Paste into a note in a code block with `dataview` language
4. The query will auto-execute and show results

