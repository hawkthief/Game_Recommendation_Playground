# Game Parameter Evaluation Vault

## Quick Start

1. **Start here:** Open `Meta/evaluation_index.md` for an overview
2. **Pick a parameter** from the Parameters folder
3. **Use the template structure** to evaluate each parameter
4. **Update the index** as you complete evaluations

## Workflow

1. Open a parameter file (e.g., `Parameters/mechanical_depth.md`)
2. Fill out the sections, especially:
   - Current Definition
   - Answer to "Why does this capture something genre labels can't?"
   - Related Parameters (use `[[parameter_name]]` to link)
3. Update the YAML frontmatter:
   - `survival_rating`: high/medium/low
   - `tags`: Add relevant tags from category_definitions
   - `status`: evaluating → keep/modify/collapse/remove
4. Move the parameter to the appropriate section in `evaluation_index.md`

## File Structure

```
/Parameters/          # Individual parameter evaluations (20 files)
/Meta/                # Meta documents
  - evaluation_index.md      # Main dashboard
  - category_definitions.md  # Tag taxonomy
  - collapse_decisions.md    # Track parameters to combine
  - parameter_template.md    # Template for reference
```

## Tips

- **Link liberally**: Use `[[other_parameter]]` to cross-reference
- **Tag consistently**: Check `category_definitions.md` for tag meanings
- **Graph view**: See relationships between parameters
- **Track collapse candidates**: Note potential combinations in `collapse_decisions.md`

## Recommended Plugins

- **Dataview**: Query and filter parameters (e.g., "show all #high-overlap parameters")
- **Tag Wrangler**: Manage your tags
- **Kanban**: Visualize parameter status
- **Outliner**: Better list handling

