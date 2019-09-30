
//__________________________________________________________________________________ for tile changing
var tile = document.getElementById("tile");
var canvas_draw = document.getElementById("myCanvas");
//__________________________________________________________________________________



//_________________________________________________________________canvas for screenshots
var canvas_solid = document.getElementById('myCanvas_solid');//set the URL from json later
var canvas_movable = document.getElementById('myCanvas_movable')
var canvas_destroyable = document.getElementById('myCanvas_destroyable')
var canvas_dangerous = document.getElementById('myCanvas_dangerous')
var canvas_gettable = document.getElementById('myCanvas_gettable')
var canvas_portal = document.getElementById('myCanvas_portal')
var canvas_usable = document.getElementById('myCanvas_usable')
var canvas_changeable = document.getElementById('myCanvas_changeable')
var canvas_ui = document.getElementById('myCanvas_ui')
//solid.src = affordances["solid"] - get info from json
//solid.onload(function()){...}
//ctx.drawImage(solid,0,0)
//make draw affordances separate function after affordances are loaded



//_________________________________for json
var mydata;
var num = 0;
//_________________________________________

var i;


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


	
//-_______________________________________________________________
function draw(x, y, z)
{
    
    if (z.getContext)
    {
        var ctx = z.getContext('2d');
        //drawing code here
        ctx.lineWidth = 3;
        ctx.fillStyle = "rgb(255, 255, 255)";
        ctx.fillRect (x, y, 16, 16);
    } 
    else 
    {
        // canvas-unsupported code here
    }
}
function draw_b(x, y, z, x_a, x_b)
{
    
    if (z.getContext)
    {
        var ctx = z.getContext('2d');
        //drawing code here
        ctx.lineWidth = 3;
        ctx.fillStyle = "rgb(0, 0, 0)";
        ctx.fillRect (x, y, x_a, x_b);
    } 
    else 
    {
        // canvas-unsupported code here
    }
}

function erase(x)
{
    if (x.getContext)
    {
        var ctx = x.getContext('2d');
        ctx.clearRect(0, 0, 256, 224)
    } 
    else 
    {
        // canvas-unsupported code here
    }
}
//_______________________________________________________________
function draw_picture(x, z)
{
    if (z.getContext)
    {
        var ctx = z.getContext('2d');   
        var img = new Image();
        //load image first ==v
        img.onload = function()
        {
            ctx.drawImage(img, 0, 0, 256, 224);
        };
        img.src = x; 
    } 
    else 
    {
        console.log("canvas not found");
    }
}

//_________________________makes all screens black by default
draw_b(0, 0, canvas_solid, 256, 224);
draw_b(0, 0, canvas_movable, 256, 224);
draw_b(0, 0, canvas_destroyable, 256, 224);
draw_b(0, 0, canvas_dangerous, 256, 224);
draw_b(0, 0, canvas_gettable, 256, 224);
draw_b(0, 0, canvas_portal, 256, 224);
draw_b(0, 0, canvas_usable, 256, 224);
draw_b(0, 0, canvas_changeable, 256, 224);
draw_b(0, 0, canvas_ui, 256, 224);
//-----------------------------------------------------------------------

//__________________________________________________enter and all the rest keys vvv

document.onkeydown = function(event)
{     
    //num = num + 1;
    var pos_x = 0;
    var pos_y = 0;
    var tiles = mydata['output']['tiles'];
    var poses = mydata['output']['tiles']['tile_' + num]['locations'];  
    switch (event.keyCode)
    {
        case 13:
            //TODO 30/09/2019:
            //generate a json (object?) without pictures but same tile_num + have solid = 0; movable = 1; etc... + no locations either + also nine black-white imahges when user is done also... image ID + tagger ID and ... upload(post) this whole file to URL/server 
            //alert('test'); //remove if needed
            //here draw all the squares around tiles
            erase(canvas_draw);
            tile.src = tiles['tile_' + num]['tile_data'];
            poses = mydata['output']['tiles']['tile_' + num]['locations']; 
             
            for(i = 0; i < Object.keys(poses).length; i++)
            {   
                pos_x = poses['location_' + i]['x'];
                pos_y = poses['location_' + i]['y'];
                draw(pos_x, pos_y, canvas_draw);
                //TODO: also draw those squares on affordances after certain key is pressed
            }
            if (num == (Object.keys(tiles).length) - 1)
            {
                alert('Out of tiles. Last tile num = ' + num);   
                break;         
            }
            else
            {
                num = num + 1;
            }
            
        break;
            //____________________vvv keypress to draw on affordances squares vvv
            //TODO: put for cycle in each or make a fucntion idk
            case 81: //q
                //draw_picture(mydata['output']['tag_images']['solid'], canvas_solid);
                var tmp = num;
                tmp = tmp - 1;
                poses = mydata['output']['tiles']['tile_' + tmp]['locations']; 
                for(i = 0; i < Object.keys(poses).length; i++)
                {   
                    pos_x = poses['location_' + i]['x'];
                    pos_y = poses['location_' + i]['y'];
                    draw(pos_x, pos_y, canvas_solid);
                    /*
                    draw_b(pos_x, pos_y, canvas_dangerous, 16, 16);
                    draw_b(pos_x, pos_y, canvas_movable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_destroyable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_ui, 16, 16);
                    draw_b(pos_x, pos_y, canvas_gettable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_portal, 16, 16);
                    draw_b(pos_x, pos_y, canvas_usable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_changeable, 16, 16);
                    */
                }
            break;
            
            case 87: //w
            var tmp = num;
            tmp = tmp - 1;
            poses = mydata['output']['tiles']['tile_' + tmp]['locations']; 
            for(i = 0; i < Object.keys(poses).length; i++)
            {   
                pos_x = poses['location_' + i]['x'];
                pos_y = poses['location_' + i]['y'];
                draw(pos_x, pos_y, canvas_movable);
                /*
                draw_b(pos_x, pos_y, canvas_solid, 16, 16);
                draw_b(pos_x, pos_y, canvas_dangerous, 16, 16);
                
                draw_b(pos_x, pos_y, canvas_destroyable, 16, 16);
                draw_b(pos_x, pos_y, canvas_ui, 16, 16);
                draw_b(pos_x, pos_y, canvas_gettable, 16, 16);
                draw_b(pos_x, pos_y, canvas_portal, 16, 16);
                draw_b(pos_x, pos_y, canvas_usable, 16, 16);
                draw_b(pos_x, pos_y, canvas_changeable, 16, 16);
                */
            }
            break;
            
            case 69: //e
                //draw_picture(mydata['output']['tag_images']['destroyable'], canvas_destroyable);
                var tmp = num;
                tmp = tmp - 1;
                poses = mydata['output']['tiles']['tile_' + tmp]['locations']; 
                for(i = 0; i < Object.keys(poses).length; i++)
                {   
                    pos_x = poses['location_' + i]['x'];
                    pos_y = poses['location_' + i]['y'];
                    draw(pos_x, pos_y, canvas_destroyable);
                    /*
                    draw_b(pos_x, pos_y, canvas_solid, 16, 16);
                    draw_b(pos_x, pos_y, canvas_dangerous, 16, 16);
                    draw_b(pos_x, pos_y, canvas_movable, 16, 16);
                    
                    draw_b(pos_x, pos_y, canvas_ui, 16, 16);
                    draw_b(pos_x, pos_y, canvas_gettable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_portal, 16, 16);
                    draw_b(pos_x, pos_y, canvas_usable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_changeable, 16, 16);
                    */
                }
            break;
            case 65: //a
                //draw_picture(mydata['output']['tag_images']['dangerous'], canvas_dangerous);
                var tmp = num;
                tmp = tmp - 1;
                poses = mydata['output']['tiles']['tile_' + tmp]['locations']; 
                for(i = 0; i < Object.keys(poses).length; i++)
                {   
                    pos_x = poses['location_' + i]['x'];
                    pos_y = poses['location_' + i]['y'];
                    draw(pos_x, pos_y, canvas_dangerous);
                    /*
                    draw_b(pos_x, pos_y, canvas_solid, 16, 16);
                    
                    draw_b(pos_x, pos_y, canvas_movable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_destroyable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_ui, 16, 16);
                    draw_b(pos_x, pos_y, canvas_gettable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_portal, 16, 16);
                    draw_b(pos_x, pos_y, canvas_usable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_changeable, 16, 16);
                    */
                }
            break;
            case 83: //s
                //draw_picture(mydata['output']['tag_images']['gettable'], canvas_gettable);
                var tmp = num;
                tmp = tmp - 1;
                poses = mydata['output']['tiles']['tile_' + tmp]['locations']; 
                for(i = 0; i < Object.keys(poses).length; i++)
                {   
                    pos_x = poses['location_' + i]['x'];
                    pos_y = poses['location_' + i]['y'];
                    draw(pos_x, pos_y, canvas_gettable);
                    /*
                    draw_b(pos_x, pos_y, canvas_solid, 16, 16);
                    draw_b(pos_x, pos_y, canvas_dangerous, 16, 16);
                    draw_b(pos_x, pos_y, canvas_movable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_destroyable), 16, 16;
                    draw_b(pos_x, pos_y, canvas_ui, 16, 16);
                    
                    draw_b(pos_x, pos_y, canvas_portal, 16, 16);
                    draw_b(pos_x, pos_y, canvas_usable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_changeable, 16, 16);
                    */
                }
            break;
            case 68: //d
                //draw_picture(mydata['output']['tag_images']['portal'], canvas_portal);
                var tmp = num;
                tmp = tmp - 1;
                poses = mydata['output']['tiles']['tile_' + tmp]['locations']; 
                for(i = 0; i < Object.keys(poses).length; i++)
                {   
                    pos_x = poses['location_' + i]['x'];
                    pos_y = poses['location_' + i]['y'];
                    draw(pos_x, pos_y, canvas_portal);
                    /*
                    draw_b(pos_x, pos_y, canvas_solid, 16, 16);
                    draw_b(pos_x, pos_y, canvas_dangerous, 16, 16);
                    draw_b(pos_x, pos_y, canvas_movable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_destroyable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_ui, 16, 16);
                    draw_b(pos_x, pos_y, canvas_gettable, 16, 16);
                    
                    draw_b(pos_x, pos_y, canvas_usable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_changeable, 16, 16);*/
                }
            break;

            case 90: //z
                //draw_picture(mydata['output']['tag_images']['usable'], canvas_usable);
                var tmp = num;
                tmp = tmp - 1;
                poses = mydata['output']['tiles']['tile_' + tmp]['locations']; 
                for(i = 0; i < Object.keys(poses).length; i++)
                {   
                    pos_x = poses['location_' + i]['x'];
                    pos_y = poses['location_' + i]['y'];
                    draw(pos_x, pos_y, canvas_usable);
                    /*draw_b(pos_x, pos_y, canvas_solid, 16, 16);
                    draw_b(pos_x, pos_y, canvas_dangerous, 16, 16);
                    draw_b(pos_x, pos_y, canvas_movable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_destroyable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_ui, 16, 16);
                    draw_b(pos_x, pos_y, canvas_gettable, 16, 16);
                    draw_b(pos_x, pos_y, canvas_portal, 16, 16);
                    
                    draw_b(pos_x, pos_y, canvas_changeable);*/
                    
                }
            break;

            case 88: //x
                //draw_picture(mydata['output']['tag_images']['changeable'], canvas_changeable);
                var tmp = num;
                tmp = tmp - 1;
                poses = mydata['output']['tiles']['tile_' + tmp]['locations']; 
                for(i = 0; i < Object.keys(poses).length; i++)
                {   
                    pos_x = poses['location_' + i]['x'];
                    pos_y = poses['location_' + i]['y'];
                    draw(pos_x, pos_y, canvas_changeable);
                    /*
                    draw_b(pos_x, pos_y, canvas_solid);
                    draw_b(pos_x, pos_y, canvas_dangerous);
                    draw_b(pos_x, pos_y, canvas_movable);
                    draw_b(pos_x, pos_y, canvas_destroyable);
                    draw_b(pos_x, pos_y, canvas_ui);
                    draw_b(pos_x, pos_y, canvas_gettable);
                    draw_b(pos_x, pos_y, canvas_portal);
                    draw_b(pos_x, pos_y, canvas_usable);
                    */
                }
            break;

            case 67: // c
                //draw_picture(mydata['output']['tag_images']['ui'], canvas_ui);
                var tmp = num;
                tmp = tmp - 1;
                poses = mydata['output']['tiles']['tile_' + tmp]['locations']; 
                for(i = 0; i < Object.keys(poses).length; i++)
                {   
                    pos_x = poses['location_' + i]['x'];
                    pos_y = poses['location_' + i]['y'];
                    draw(pos_x, pos_y, canvas_ui);
                    /*
                    draw_b(pos_x, pos_y, canvas_solid);
                    draw_b(pos_x, pos_y, canvas_dangerous);
                    draw_b(pos_x, pos_y, canvas_movable);
                    draw_b(pos_x, pos_y, canvas_destroyable);
                    
                    draw_b(pos_x, pos_y, canvas_gettable);
                    draw_b(pos_x, pos_y, canvas_portal);
                    draw_b(pos_x, pos_y, canvas_usable);
                    draw_b(pos_x, pos_y, canvas_changeable);
                    */
                }
            break;

            case 27: // ECS
            var tmp = num;
            tmp = tmp - 1;
            poses = mydata['output']['tiles']['tile_' + tmp]['locations']; 
            for(i = 0; i < Object.keys(poses).length; i++)
            {   
                pos_x = poses['location_' + i]['x'];
                pos_y = poses['location_' + i]['y'];
                draw_b(pos_x, pos_y, canvas_solid, 16, 16);
                draw_b(pos_x, pos_y, canvas_dangerous, 16, 16);
                draw_b(pos_x, pos_y, canvas_movable, 16, 16);
                draw_b(pos_x, pos_y, canvas_destroyable, 16, 16);
                draw_b(pos_x, pos_y, canvas_ui, 16, 16);
                draw_b(pos_x, pos_y, canvas_gettable, 16, 16);
                draw_b(pos_x, pos_y, canvas_portal, 16, 16);
                draw_b(pos_x, pos_y, canvas_usable, 16, 16);
                draw_b(pos_x, pos_y, canvas_changeable, 16, 16);
            }  
            break;
            
            
    }
}
