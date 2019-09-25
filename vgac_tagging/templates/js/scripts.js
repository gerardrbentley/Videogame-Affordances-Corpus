var tile = document.getElementById("tile");
var canvas_draw = document.getElementById("myCanvas");

var mydata;
var num = 0;

(function() 
{
    // Load the script
    var script = document.createElement("SCRIPT");
    script.src = 'https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js';
    script.type = 'text/javascript';
    script.onload = function() 
    {
        var $ = window.jQuery;
        // Use $ here...
        $.getJSON("example_data.json", function(json) 
        {
            mydata = json;
            //console.log(json); // this will show the info it in firebug console
        });
    };
    document.getElementsByTagName("head")[0].appendChild(script);
})();

//TODO: can host it on a real URL; find one online 4 free 
//TODO: draw a square around current tile; on enter refresh
// get coordinates and draw a square on top
var x_pos;
var y_pos;

function draw(x, y)
{
    
    if (canvas_draw.getContext)
    {
        var ctx = canvas_draw.getContext('2d');
        //drawing code here
        ctx.lineWidth = 1;
        ctx.fillStyle = "rgb(0,0,0)";
        ctx.strokeRect (x, y, 16, 16);
    } 
    else 
    {
        // canvas-unsupported code here
    }
}
var count = 0;
document.onkeydown = function(event)
{     
    //num = num + 1;
    var tiles = mydata['output']['tiles'];  
    var positions = mydata['output']['tiles'][]
    switch (event.keyCode)
    {
        case 13:
            //alert('test'); //remove if needed
            for (i = 0; i < )
            draw(0, 0);//here draw all the squares around tiles
            tile.src = tiles['tile_' + num]['tile_data'];
            if (num == (Object.keys(tiles).length - 1))
            {
                alert('Out of tiles. Last tile num = ' + num);   
                break;         
            }
            else
            {
                num = num + 1;
            }
            break;
    }
}
