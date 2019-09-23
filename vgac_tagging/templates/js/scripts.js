var tile = document.getElementById("tile");
var tile2 = document.getElementById("tile2");

var tileList = [tile, tile2];

/* doesn't work yet */
document.addEventListener("keydown", event => {
    for (i = tileList.length; i > 0; i--){
        if (event.keyCode === 13){
            tileList[i].style.visibility = "hidden";
        }
    }
});