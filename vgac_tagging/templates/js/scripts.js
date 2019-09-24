var tile = document.getElementById("tile");
var tile2 = document.getElementById("tile2");
var tile3 = document.getElementById("tile3");
var tile4 = document.getElementById("tile4");
var tile5 = document.getElementById("tile5");
var tile6 = document.getElementById("tile6");
var tile7 = document.getElementById("tile7");
var tile8 = document.getElementById("tile8");
var tile9 = document.getElementById("tile9");
var tile10 = document.getElementById("tile10");
var tile11 = document.getElementById("tile11");
var tile12 = document.getElementById("tile12");
var tile13 = document.getElementById("tile13");
var tile14 = document.getElementById("tile14");

var tileList = [tile, tile2, tile3, tile4, tile5, tile6, tile7, tile8, tile9, tile10, tile11, tile12, tile13, tile14];
var _ = tileList.length - 1;
/* doesn't work yet */
/*document.addEventListener("keydown", event => 
{
    for (i = tileList.length; i < tileList.length; i++)
    {
        if (event.keyCode === 13)
        {
            alert('Hello');
        }
    }
});*/
document.onkeydown = function(event)
{
        
    switch (event.keyCode)
    {
        case 13:
            alert('fuck');
            tileList[_].style.visibility = "hidden";
            _ = _ -1;
            break;
    }
}