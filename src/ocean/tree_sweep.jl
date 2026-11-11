include("simulation_sweep.jl")
using Graphs

# Iterative pre-order DFS on a directed tree (edges parent -> child)
function dfs_preorder(tree, root)
    order = Int[]
    stack = [root]
    while !isempty(stack)
        v = pop!(stack)
        push!(order, v)
        # push children in reverse so the smallest-index child is visited first
        for w in reverse(outneighbors(tree, v))
            push!(stack, w)
        end
    end
    return order
end

# BFS tree over a grid of `lengths` from the coordinate `root`: the DFS visiting order and each vertex's parent (0 for the root)
function sweep_tree(lengths; root = lengths .÷ 2 .+ 1)
    root_vertex = LinearIndices(lengths)[root...]
    tree = bfs_tree(Graphs.SimpleGraphs.grid(collect(lengths)), root_vertex)
    parents = [isempty(inneighbors(tree, v)) ? 0 : only(inneighbors(tree, v)) for v in vertices(tree)]
    return dfs_preorder(tree, root_vertex), parents
end

"""
    tree_plan(prefix, swept; root = nothing)

The runs of a tree sweep over the NamedTuple or keyword pairs `swept`, parents first, as (`name`, `point`, `parent`) with `parent` the
parent's run name (`nothing` for the root). `root` is the coordinate of the first run, by default `n ÷ 2 + 1` along each axis.
"""
function tree_plan(prefix, swept; root = nothing)
    axes_values = Tuple(values(swept))
    lengths = length.(axes_values)
    order, parents = sweep_tree(lengths; root = something(root, lengths .÷ 2 .+ 1))
    coordinates = CartesianIndices(lengths)
    point(v) = NamedTuple{Tuple(keys(swept))}(ntuple(i -> axes_values[i][coordinates[v][i]], length(lengths)))
    name(v) = run_name(prefix, point(v))
    ancestor(v) = parents[v] == 0 ? nothing : parents[v]
    function node(v)
        p, g = ancestor(v), isnothing(ancestor(v)) ? nothing : ancestor(ancestor(v))
        return (; name = name(v), point = point(v), parent = isnothing(p) ? nothing : name(p), grandparent = isnothing(g) ? nothing : name(g),
                parent_point = isnothing(p) ? nothing : point(p), grandparent_point = isnothing(g) ? nothing : point(g))
    end
    return [node(v) for v in order]
end

usable_spin_up(name) = is_finished(name) && !blew_up(name)

# Coordinate in which T is taken as linear along an axis: T_day itself, since T − T_night scales with ΔT, and the logarithm of the others
predictor_coordinate(axis, value) = axis == :T_day ? float(value) : log(float(value))

# Weight of T_parent − T_grandparent in the predictor (Keller 1977), or nothing unless the three runs lie in order along one axis in `axes`
function predictor_weight(node, axes)
    isnothing(node.grandparent) && return nothing
    differs(a, b) = [k for k in keys(a) if a[k] != b[k]]
    step, previous = differs(node.point, node.parent_point), differs(node.parent_point, node.grandparent_point)
    (length(step) == 1 && step == previous && only(step) in axes) || return nothing
    a = only(step)
    x, x_p, x_g = predictor_coordinate.(a, (node.point[a], node.parent_point[a], node.grandparent_point[a]))
    w = (x - x_p) / (x_p - x_g)
    return w > 0 ? w : nothing
end

last_frame(ds, name) = (n = size(ds[name], 4); Array(ds[name][:, :, :, n]))

# Start state T_parent + w (T_parent − T_grandparent) clamped to [T_min, T_max], or nothing when the clamp removes too much or the mean profile has a large inversion
function predicted_temperature(T_parent, T_grandparent, w; T_min, T_max, max_removed = 0.5, max_inversion = 0.05)
    raw = T_parent .+ w .* (T_parent .- T_grandparent)
    predicted = clamp.(raw, T_min, T_max)
    sqrt(mean((raw .- predicted) .^ 2)) > max_removed * sqrt(mean((T_parent .- T_min) .^ 2)) && return nothing
    profile = vec(mean(predicted; dims = (1, 2)))
    return minimum(diff(profile)) >= -max_inversion * (T_max - T_min) ? predicted : nothing
end

"""
    predicted_donor(name, parent, grandparent, w; T_min, T_max)

Write a run `name` to warm-start from, holding the predicted T and the parent's u, v and grid, and return its name; returns `parent` if the prediction is rejected.
"""
function predicted_donor(name, parent, grandparent, w; T_min, T_max)
    is_finished(name) && isfile(joinpath(directory, "$(name).nc")) && return name
    T_parent, u, v = NCDataset(ds -> (last_frame(ds, "T"), last_frame(ds, "u"), last_frame(ds, "v")), joinpath(directory, "$(parent).nc"))
    T_grandparent = NCDataset(ds -> last_frame(ds, "T"), joinpath(directory, "$(grandparent).nc"))
    T = predicted_temperature(T_parent, T_grandparent, w; T_min, T_max)
    if isnothing(T)
        @warn "Predictor rejected for $(name): warm-starting from a copy of $(parent)"
        return parent
    end
    path, partial = joinpath(directory, "$(name).nc"), joinpath(directory, "$(name).nc.partial")
    NCDataset(partial, "c") do ds
        defDim(ds, "time", 1)
        for (variable, data) in (("T", T), ("u", u), ("v", v))
            dims = ["$(variable)_$(d)" for d in ("x", "y", "z")]
            foreach((d, n) -> defDim(ds, d, n), dims, size(data))
            defVar(ds, variable, reshape(data, size(data)..., 1), (dims..., "time"))
        end
    end
    mv(partial, path; force = true)
    data = read_parameters(parent)
    data["run"] = Dict("name" => name)
    data["predictor"] = Dict{String, Any}("parent" => parent, "grandparent" => grandparent, "weight" => w)
    open(io -> TOML.print(io, data; sorted = true), parameter_file(name), "w")
    return name
end

function run_status(name, spin_up_name)
    blew_up(spin_up_name) && return "spin-up blew up"
    blew_up(name) && return "main run blew up"
    return is_finished(name) ? "ok" : "unfinished"
end

# usable_spin_up for another worker's run: false while its parameter file is being rewritten
ready_spin_up(name) = try usable_spin_up(name) catch; false end
failed_spin_up(name) = try blew_up(name) catch; false end

"""
    tree_sweep(base; prefix = "tree", point_settings = Returns((;)), spin_up = Returns(SPIN_UP), root = nothing, plan_only = false, stages = :all, swept...)

Run every combination of the swept values as a `simulation` of a 2° spin-up then a main stage, depth-first through a BFS tree of the
parameter grid. Each spin-up warm-starts from its parent's (one step along one axis), the root's from rest, and each main stage from its
own spin-up. `point_settings(point)` gives each run's extra keywords and `spin_up(point, warm_started)` its `low_res_start`. Rerunning skips finished
stages and resumes cut-off ones. A run whose parent's spin-up is missing or blew up is blocked, and the rest of the tree carries on.
`plan_only` prints the tree without running it. Along the `predictor_axes`, a run in line with its parent and grandparent starts from their
extrapolated T rather than a copy of its parent (see `predicted_donor`). `stages = :spin_ups` and `:mains` split the sweep between two processes
started together: one walks the tree running spin-ups, the other runs main stages as their spin-ups finish (see `main_stage_worker`).
Returns each run's name and status.
"""
function tree_sweep(base; prefix = "tree", point_settings = Returns((;)), spin_up = Returns(SPIN_UP), root = nothing, plan_only = false, predictor_axes = (),
                    stages = :all, swept...)
    stages in (:all, :spin_ups, :mains) || error("stages must be :all, :spin_ups or :mains, got $(repr(stages))")
    plan = tree_plan(prefix, swept; root)
    if plan_only
        foreach(node -> println(node.name, "  <-  ", something(node.parent, "rest"), isnothing(predictor_weight(node, predictor_axes)) ? "" : "  (predictor)"), plan)
        return [node.name => "planned" for node in plan]
    end

    function run_point(node, which)
        spin_up_name = node.name * "_spinup"
        parent_spin_up = isnothing(node.parent) ? nothing : node.parent * "_spinup"
        params = merge(SWEEP_STORAGE, base, node.point, point_settings(node.point), (; name = node.name))
        minimum(get(params, :resolution, 1.0)) < 1 && !haskey(params, :save_last) && (params = merge(params, (; save_last = HIGH_RES_SAVE_LAST)))
        w = :spin_up in which ? predictor_weight(node, predictor_axes) : nothing
        donor = isnothing(w) || is_finished(spin_up_name) || !usable_spin_up(node.grandparent * "_spinup") ? parent_spin_up :
                predicted_donor(spin_up_name * "_predictor", parent_spin_up, node.grandparent * "_spinup", w; T_min = params.T_night, T_max = params.T_day)
        simulation(params, spin_up(node.point, !isnothing(parent_spin_up)), donor; spin_up_name, stages = which)
        return :main in which ? run_status(node.name, spin_up_name) : blew_up(spin_up_name) ? "spin-up blew up" : is_finished(spin_up_name) ? "spin-up ok" : "spin-up unfinished"
    end
    function attempt(node, which)
        try
            run_point(node, which)
        catch err
            @error "Run $(node.name) failed" exception = (err, catch_backtrace())
            "error"
        end
    end

    marker = joinpath(directory, "$(prefix)_spin_up_worker_finished")    # tells the main-stage worker that no more spin-ups are coming
    stages == :mains && return main_stage_worker(plan, node -> attempt(node, (:main,)); marker)
    stages == :spin_ups && rm(marker; force = true)
    statuses = Pair{String, String}[]
    which = stages == :all ? (:spin_up, :main) : (:spin_up,)
    for (i, node) in enumerate(plan)
        parent_spin_up = isnothing(node.parent) ? nothing : node.parent * "_spinup"
        @info "Tree sweep: run $(i) of $(length(plan)), $(node.name), warm start $(something(parent_spin_up, "from rest"))"
        if !isnothing(parent_spin_up) && !usable_spin_up(parent_spin_up)
            @error "Blocked $(node.name): $(parent_spin_up) has no usable final state"
            push!(statuses, node.name => "blocked by $(node.parent)")
            continue
        end
        push!(statuses, node.name => attempt(node, which))
    end
    stages == :spin_ups && touch(marker)
    foreach(s -> @info("$(s.first): $(s.second)"), statuses)
    return statuses
end

"""
    main_stage_worker(plan, run_main; marker = nothing, poll = 300, patience = 12 * 3600)

Run `run_main(node)` for each node of `plan` once its spin-up has finished, in plan order, while another process runs the spin-ups. Waits `poll`
seconds when nothing is ready, and stops when every spin-up has finished or is blocked by a parent that blew up, when nothing is ready and the
file `marker` exists (the spin-up worker has finished, so spin-ups that errored will never come), or after `patience` seconds with nothing
becoming ready (the spin-up worker has probably stopped).
"""
function main_stage_worker(plan, run_main; marker = nothing, poll = 300, patience = 12 * 3600)
    nodes = Dict(node.name => node for node in plan)
    blocked(node) = !isnothing(node.parent) && (failed_spin_up(node.parent * "_spinup") || blocked(nodes[node.parent]))
    statuses = Pair{String, String}[]
    done = Set{String}()
    idle = 0
    started = time()    # a marker older than this is left from an earlier run
    while true
        spin_ups_over = !isnothing(marker) && isfile(marker) && mtime(marker) > started    # read before the scan, so the last spin-ups are seen
        ready = findfirst(node -> !(node.name in done) && ready_spin_up(node.name * "_spinup"), plan)
        if !isnothing(ready)
            node = plan[ready]
            @info "Main-stage worker: $(node.name) ($(length(done) + 1) of $(length(plan)))"
            push!(statuses, node.name => run_main(node))
            push!(done, node.name)
            idle = 0
            continue
        end
        waiting = [node.name for node in plan if !(node.name in done) && !failed_spin_up(node.name * "_spinup") && !blocked(node)]
        isempty(waiting) && break
        if spin_ups_over
            @warn "Main-stage worker: the spin-up worker has finished; $(length(waiting)) points have no spin-up, e.g. $(first(waiting))"
            break
        end
        if idle >= patience
            @error "Main-stage worker: no spin-up finished in $(round(idle / 3600, digits = 1)) h; stopping with $(length(waiting)) points left, e.g. $(first(waiting))"
            break
        end
        @info "Main-stage worker: waiting for $(length(waiting)) spin-ups, next $(first(waiting))"
        sleep(poll)
        idle += poll
    end
    for node in plan
        node.name in done && continue
        push!(statuses, node.name => (failed_spin_up(node.name * "_spinup") ? "spin-up blew up" : blocked(node) ? "blocked" : "no spin-up"))
    end
    foreach(s -> @info("$(s.first): $(s.second)"), statuses)
    return statuses
end
