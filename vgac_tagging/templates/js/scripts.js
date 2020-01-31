var c = document.getElementById("canvas_tile");
var ctx = c.getContext("2d");
ctx.beginPath();
ctx.arc(289, 289, 288, 0, 180);
ctx.strokeStyle = "rgb(113, 244, 244)";
ctx.stroke();
ctx.fillStyle = "rgb(113, 244, 244)";
ctx.fill();

function hideSVG(id) 
{
    var style = document.getElementById(id).style.display;
    if(style === "none")
      document.getElementById(id).style.display = "block";
    else
      document.getElementById(id).style.display = "none";
    //or to hide the all svg
    //document.getElementById("mySvg").style.display = "none";
  }

var b_solid = document.getElementById("b_solid");
b_solid.style.backgroundColor = "#ff66ff";
b_solid.style.fontSize = "x-large";
b_solid.style.fontVariant = "small-caps";


var b_movable = document.getElementById("b_movable");
b_movable.style.backgroundColor = "#ff66ff";
b_movable.style.fontSize = "x-large";
b_movable.style.fontVariant = "small-caps";

var b_destroyable = document.getElementById("b_destroyable");
b_destroyable.style.backgroundColor = "#ff66ff";
b_destroyable.style.fontSize = "x-large";
b_destroyable.style.fontVariant = "small-caps";

var b_dangerous = document.getElementById("b_dangerous");
b_dangerous.style.backgroundColor = "#ff66ff";
b_dangerous.style.fontSize = "x-large";
b_dangerous.style.fontVariant = "small-caps";

var b_gettable = document.getElementById("b_gettable");
b_gettable.style.backgroundColor = "#ff66ff";
b_gettable.style.fontSize = "x-large";
b_gettable.style.fontVariant = "small-caps";

var b_portal = document.getElementById("b_portal");
b_portal.style.backgroundColor = "#ff66ff";
b_portal.style.fontSize = "x-large";
b_portal.style.fontVariant = "small-caps";

var b_usable = document.getElementById("b_usable");
b_usable.style.backgroundColor = "#ff66ff";
b_usable.style.fontSize = "x-large";
b_usable.style.fontVariant = "small-caps";

var b_changeable = document.getElementById("b_changeable");
b_changeable.style.backgroundColor = "#ff66ff";
b_changeable.style.fontSize = "x-large";
b_changeable.style.fontVariant = "small-caps";

var b_ui = document.getElementById("b_ui");
b_ui.style.backgroundColor = "#ff66ff";
b_ui.style.fontSize = "x-large";
b_ui.style.fontVariant = "small-caps";

var b_permeable = document.getElementById("b_permeable");
b_permeable.style.backgroundColor = "#ff66ff";
b_permeable.style.fontSize = "x-large";
b_permeable.style.fontVariant = "small-caps";

var b_save = document.getElementById("b_save");
b_save.style.backgroundColor = "green";
b_save.style.fontSize = "x-large";
b_save.style.fontVariant = "small-caps";

var b_comment = document.getElementById("b_comment");
b_comment.style.backgroundColor = "grey";
b_comment.style.fontSize = "x-large";
b_comment.style.fontVariant = "small-caps";

var b_reset = document.getElementById("b_reset");
b_reset.style.backgroundColor = "red";
b_reset.style.fontSize = "x-large";
b_reset.style.fontVariant = "small-caps";


