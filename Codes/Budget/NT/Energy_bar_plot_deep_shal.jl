# ============================================================
# Bar plot of time-averaged energy budget terms
# Shallow (left panel) and Deep (right panel)
# ============================================================
# Assumes sh and dp NamedTuples are already computed from the main script


sc = 1e8   # same scaling as time series


# --- Time-average each term ---
function time_mean_terms(d, sc)
    return (
        Conv     = mean(d.Conv)     * sc,
        FDiv     = mean(d.FDiv)     * sc,
        SP_H     = mean(d.SP_H)     * sc,
        SP_V     = mean(d.SP_V)     * sc,
        BP       = mean(d.BP)       * sc,
        A        = mean(d.A)        * sc,
        ET       = mean(d.ET)       * sc,
        Residual = mean(d.Residual) * sc,
    )
end


sh_mean = time_mean_terms(sh, sc)
dp_mean = time_mean_terms(dp, sc)


# --- Labels and values (same order for both panels) ---
labels = [
    "⟨R⟩ Residual (D)",
    "⟨A⟩ Advection",
    "⟨Pᵦ⟩ Buoyancy prod.",
    "⟨Pₛᵛ⟩ Vert. shear prod.",
    "⟨Pₛᴴ⟩ Horiz. shear prod.",
    "⟨∂E/∂t⟩ Tendency",
    "⟨∇·F⟩ Flux divergence",
    "⟨C⟩ Conversion",
]


function extract_vals(d)
    return [
        d.Residual,
        d.A,
        d.BP,
        d.SP_V,
        d.SP_H,
        d.ET,
        d.FDiv,
        d.Conv,
    ]
end


sh_vals = extract_vals(sh_mean)
dp_vals = extract_vals(dp_mean)


n = length(labels)
y_pos = collect(1:n)   # bar positions


# --- Bar colors: positive = warm red, negative = cool blue ---
function bar_color(v)
    return v >= 0 ? RGBf(0.75, 0.15, 0.15) : RGBf(0.20, 0.40, 0.75)
end


sh_colors = bar_color.(sh_vals)
dp_colors = bar_color.(dp_vals)


# ============================================================
# Figure
# ============================================================
fig = Figure(resolution=(900, 480), backgroundcolor=:white,
             fonts=(; regular=FONT))


bar_theme = (
    backgroundcolor   = :white,
    xgridcolor        = grid_col,
    ygridcolor        = RGBAf(0,0,0,0),   # no horizontal grid
    xgridwidth        = 0.6,
    xtickcolor        = tick_col,
    ytickcolor        = RGBAf(0,0,0,0),
    xticklabelcolor   = tick_col,
    yticklabelcolor   = RGBf(0.10, 0.10, 0.10),
    xlabelcolor       = RGBf(0.10, 0.10, 0.10),
    titlecolor        = RGBf(0.05, 0.05, 0.05),
    titlesize         = 14,
    titlealign        = :left,
    xlabelsize        = 12,
    yticklabelsize    = 11,
    xticklabelsize    = 10,
    spinewidth        = 0.8,
    topspinevisible   = false,
    rightspinevisible = false,
    leftspinecolor    = tick_col,
    bottomspinecolor  = tick_col,
    titlefont         = FONT,
    xlabelfont        = FONT,
    ylabelfont        = FONT,
    xticklabelfont    = FONT,
    yticklabelfont    = FONT,
    yticks            = (y_pos, labels),
    ytickalign        = 1,
)


# --- Shallow panel (left) ---
ax_sh = Axis(fig[1, 1];
    title  = "(a)  Shallow region  (H < $(Int(DEPTH_THRESHOLD)) m)",
    xlabel = "Energy rate  [×10⁻⁸ W/kg]",
    bar_theme...)


# --- Deep panel (right) ---
ax_dp = Axis(fig[1, 2];
    title  = "(b)  Deep region  (H ≥ $(Int(DEPTH_THRESHOLD)) m)",
    xlabel = "Energy rate  [×10⁻⁸ W/kg]",
    yticklabelsvisible = false,   # hide labels on right panel (shared y-axis)
    bar_theme...)


for (ax, vals, cols) in [(ax_sh, sh_vals, sh_colors), (ax_dp, dp_vals, dp_colors)]


    # Zero reference line
    vlines!(ax, [0.0]; color=RGBAf(0, 0, 0, 0.4), linewidth=0.9, linestyle=:dash)


    # Bars
    barplot!(ax, y_pos, vals;
        direction   = :x,
        color       = cols,
        bar_labels  = [@sprintf("%.3f", v) for v in vals],
        label_size  = 10,
        label_font  = FONT,
        label_color = RGBf(0.15, 0.15, 0.15),
        gap         = 0.25,
    )
end


# Link y-axes so bar positions align
linkyaxes!(ax_sh, ax_dp)


colgap!(fig.layout, 1, 8)


outpath = joinpath(FIGDIR, "Budget_BarPlot_TimeAvg_DepthSplit.png")
save(outpath, fig, px_per_unit=2)
println("Bar plot saved → $outpath")
display(fig)








