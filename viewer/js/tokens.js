/* The paper's figure grammar, as numbers: colours, type and strokes.
 *
 * These mirror the paper's figure registry (registry revision 3) so a
 * still or a video frame exported from the viewer obeys the same rules as the
 * paper's figures: Arial at one size, strokes of at least 1 pt, colours from
 * the registry only, white paper. Sizes are in points; the scene converts
 * nothing but world units. A check outside this repository pins every value
 * here to the registry, so the two cannot drift apart silently.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else (root.PEV = root.PEV || {}).tokens = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  return {
    REGISTRY_VERSION: 3,
    // colours
    TEXT: "#000000",          // all text, outlines, the order never received
    WHITE: "#FFFFFF",         // paper
    ZERO: "#8F8B85",          // warm grey: frames, radio links, lost and destroyed things, birds and civilian drones
    FILL: "#F4F1DE",          // eggshell: the defended region
    GRID: "#E8E4D0",          // never-swept ground (explore layer only)
    DEF: "#3D405B",           // defenders (style.TEAM["blue"]["color"])
    DEF_FILL: "#ECECF1",      // sensor wedges (style.TEAM["blue"]["fill"])
    INT: "#B34E36",           // intruders (style.TEAM["red"]["color"])
    INT_FILL: "#FDE8E2",      // (style.TEAM["red"]["fill"])
    // type and strokes, in points
    FONT: "Arial",
    FS: 7.0,
    LW_MIN: 1.0,
    LW_RULE: 1.0,
    LW_DATA: 1.25,
    // marks, in points (matplotlib markersize: a circle's diameter, a triangle's height)
    MS: 4.0,
    MS_COMMANDER: 7.5,
    // dash patterns, in points (matplotlib (offset, (on, off)) at 1 pt)
    DASH_PURSUIT: [3, 2],
    DASH_ORDER: [1, 1.5],
    // the print widths a figure may take (mm) and its height cap
    WIDTHS_MM: [88, 120, 180],
    H_MAX_MM: 170,
  };
});
