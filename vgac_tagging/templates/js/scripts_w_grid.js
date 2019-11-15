/*
─────────────────────────────────────────────────────────────
─██████████████─██████████████─████████████───██████████████─
─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░████─██░░░░░░░░░░██─
─██████░░██████─██░░██████░░██─██░░████░░░░██─██░░██████░░██─
─────██░░██─────██░░██──██░░██─██░░██──██░░██─██░░██──██░░██─
─────██░░██─────██░░██──██░░██─██░░██──██░░██─██░░██──██░░██─
─────██░░██─────██░░██──██░░██─██░░██──██░░██─██░░██──██░░██─
─────██░░██─────██░░██──██░░██─██░░██──██░░██─██░░██──██░░██─
─────██░░██─────██░░██──██░░██─██░░██──██░░██─██░░██──██░░██─
─────██░░██─────██░░██████░░██─██░░████░░░░██─██░░██████░░██─
─────██░░██─────██░░░░░░░░░░██─██░░░░░░░░████─██░░░░░░░░░░██─
─────██████─────██████████████─████████████───██████████████─
─────────────────────────────────────────────────────────────
*/
//second week of november - tagging event 15th
//DONE:prevent mouse or trap mouse css
//DONE: 24/10/19: event called oncontext menu -> capture on canvas element; check riot
//DONE: 17/10/19: make image bigger on hover DONE: change colors of drawing
//DONE: 17/10/19: erase on rightclick, make reset go back to last state of current tile
//DONE: 15/10/19: make draw with mouse on affordances images: one click draw white other click draw black
//DONE: 7/12/19: after json ready: url with post -> browser sending data to the server (look at riot screenshot), then reload the page to renew json or use ajax get more json
//DONE: 30/09/19: generate a json (object?) without pictures but same tile_num + solid = 0; movable = 1; etc... + no locations either + also nine black-white imahges when user is done also... image ID + tagger ID and ... upload(post) this whole file to URL/server


/*
────────────────────────────────────────────────────────────────────────────────────────────────────────────────
─██████████████─██████████████─████████████████──────██████████████─██████████─██████─────────██████████████────
─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░░░██──────██░░░░░░░░░░██─██░░░░░░██─██░░██─────────██░░░░░░░░░░██────
─██░░██████████─██░░██████░░██─██░░████████░░██──────██████░░██████─████░░████─██░░██─────────██░░██████████────
─██░░██─────────██░░██──██░░██─██░░██────██░░██──────────██░░██───────██░░██───██░░██─────────██░░██────────────
─██░░██████████─██░░██──██░░██─██░░████████░░██──────────██░░██───────██░░██───██░░██─────────██░░██████████────
─██░░░░░░░░░░██─██░░██──██░░██─██░░░░░░░░░░░░██──────────██░░██───────██░░██───██░░██─────────██░░░░░░░░░░██────
─██░░██████████─██░░██──██░░██─██░░██████░░████──────────██░░██───────██░░██───██░░██─────────██░░██████████────
─██░░██─────────██░░██──██░░██─██░░██──██░░██────────────██░░██───────██░░██───██░░██─────────██░░██────────────
─██░░██─────────██░░██████░░██─██░░██──██░░██████────────██░░██─────████░░████─██░░██████████─██░░██████████────
─██░░██─────────██░░░░░░░░░░██─██░░██──██░░░░░░██────────██░░██─────██░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░██────
─██████─────────██████████████─██████──██████████────────██████─────██████████─██████████████─██████████████────
────────────────────────────────────────────────────────────────────────────────────────────────────────────────
───────────────────────────────────────────────────────────────────────────────────────────────────
─██████████████─██████──██████─██████████████─██████──────────██████─██████████████─██████████████─
─██░░░░░░░░░░██─██░░██──██░░██─██░░░░░░░░░░██─██░░██████████──██░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─
─██░░██████████─██░░██──██░░██─██░░██████░░██─██░░░░░░░░░░██──██░░██─██░░██████████─██░░██████████─
─██░░██─────────██░░██──██░░██─██░░██──██░░██─██░░██████░░██──██░░██─██░░██─────────██░░██─────────
─██░░██─────────██░░██████░░██─██░░██████░░██─██░░██──██░░██──██░░██─██░░██─────────██░░██████████─
─██░░██─────────██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██──██░░██──██░░██─██░░██──██████─██░░░░░░░░░░██─
─██░░██─────────██░░██████░░██─██░░██████░░██─██░░██──██░░██──██░░██─██░░██──██░░██─██░░██████████─
─██░░██─────────██░░██──██░░██─██░░██──██░░██─██░░██──██░░██████░░██─██░░██──██░░██─██░░██─────────
─██░░██████████─██░░██──██░░██─██░░██──██░░██─██░░██──██░░░░░░░░░░██─██░░██████░░██─██░░██████████─
─██░░░░░░░░░░██─██░░██──██░░██─██░░██──██░░██─██░░██──██████████░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─
─██████████████─██████──██████─██████──██████─██████──────────██████─██████████████─██████████████─
───────────────────────────────────────────────────────────────────────────────────────────────────*/


//__________________________________________________________________________________

var tile = document.getElementById("tile");
var canvas_draw = document.getElementById("myCanvas");
//__________________________________________________________________________________
/*
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
─██████████████─██████████─██████──────────██████─██████──██████─██████─────────██████████████─██████████████─██████████████─
─██░░░░░░░░░░██─██░░░░░░██─██░░██████████████░░██─██░░██──██░░██─██░░██─────────██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─
─██░░██████████─████░░████─██░░░░░░░░░░░░░░░░░░██─██░░██──██░░██─██░░██─────────██░░██████░░██─██████░░██████─██░░██████████─
─██░░██───────────██░░██───██░░██████░░██████░░██─██░░██──██░░██─██░░██─────────██░░██──██░░██─────██░░██─────██░░██─────────
─██░░██████████───██░░██───██░░██──██░░██──██░░██─██░░██──██░░██─██░░██─────────██░░██████░░██─────██░░██─────██░░██████████─
─██░░░░░░░░░░██───██░░██───██░░██──██░░██──██░░██─██░░██──██░░██─██░░██─────────██░░░░░░░░░░██─────██░░██─────██░░░░░░░░░░██─
─██████████░░██───██░░██───██░░██──██████──██░░██─██░░██──██░░██─██░░██─────────██░░██████░░██─────██░░██─────██░░██████████─
─────────██░░██───██░░██───██░░██──────────██░░██─██░░██──██░░██─██░░██─────────██░░██──██░░██─────██░░██─────██░░██─────────
─██████████░░██─████░░████─██░░██──────────██░░██─██░░██████░░██─██░░██████████─██░░██──██░░██─────██░░██─────██░░██████████─
─██░░░░░░░░░░██─██░░░░░░██─██░░██──────────██░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██──██░░██─────██░░██─────██░░░░░░░░░░██─
─██████████████─██████████─██████──────────██████─██████████████─██████████████─██████──██████─────██████─────██████████████─
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
─██████──████████─██████████████─████████──████████─██████████████───██████████████─██████████████─████████████████───████████████───
─██░░██──██░░░░██─██░░░░░░░░░░██─██░░░░██──██░░░░██─██░░░░░░░░░░██───██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░░░██───██░░░░░░░░████─
─██░░██──██░░████─██░░██████████─████░░██──██░░████─██░░██████░░██───██░░██████░░██─██░░██████░░██─██░░████████░░██───██░░████░░░░██─
─██░░██──██░░██───██░░██───────────██░░░░██░░░░██───██░░██──██░░██───██░░██──██░░██─██░░██──██░░██─██░░██────██░░██───██░░██──██░░██─
─██░░██████░░██───██░░██████████───████░░░░░░████───██░░██████░░████─██░░██──██░░██─██░░██████░░██─██░░████████░░██───██░░██──██░░██─
─██░░░░░░░░░░██───██░░░░░░░░░░██─────████░░████─────██░░░░░░░░░░░░██─██░░██──██░░██─██░░░░░░░░░░██─██░░░░░░░░░░░░██───██░░██──██░░██─
─██░░██████░░██───██░░██████████───────██░░██───────██░░████████░░██─██░░██──██░░██─██░░██████░░██─██░░██████░░████───██░░██──██░░██─
─██░░██──██░░██───██░░██───────────────██░░██───────██░░██────██░░██─██░░██──██░░██─██░░██──██░░██─██░░██──██░░██─────██░░██──██░░██─
─██░░██──██░░████─██░░██████████───────██░░██───────██░░████████░░██─██░░██████░░██─██░░██──██░░██─██░░██──██░░██████─██░░████░░░░██─
─██░░██──██░░░░██─██░░░░░░░░░░██───────██░░██───────██░░░░░░░░░░░░██─██░░░░░░░░░░██─██░░██──██░░██─██░░██──██░░░░░░██─██░░░░░░░░████─
─██████──████████─██████████████───────██████───────████████████████─██████████████─██████──██████─██████──██████████─████████████───
────────────────────────────────────────────────────────────────────────────────
─██████████████─████████████████───██████████████─██████████████─██████████████─
─██░░░░░░░░░░██─██░░░░░░░░░░░░██───██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─
─██░░██████░░██─██░░████████░░██───██░░██████████─██░░██████████─██░░██████████─
─██░░██──██░░██─██░░██────██░░██───██░░██─────────██░░██─────────██░░██─────────
─██░░██████░░██─██░░████████░░██───██░░██████████─██░░██████████─██░░██████████─
─██░░░░░░░░░░██─██░░░░░░░░░░░░██───██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─
─██░░██████████─██░░██████░░████───██░░██████████─██████████░░██─██████████░░██─
─██░░██─────────██░░██──██░░██─────██░░██─────────────────██░░██─────────██░░██─
─██░░██─────────██░░██──██░░██████─██░░██████████─██████████░░██─██████████░░██─
─██░░██─────────██░░██──██░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─
─██████─────────██████──██████████─██████████████─██████████████─██████████████─
────────────────────────────────────────────────────────────────────────────────




*/
//__________________________________________________________________________________
function simulate(el, keyCode, key)
{
    let evtDown = new KeyboardEvent('keydown',
    {
        keyCode: keyCode,
        which: keyCode,
        code: key,
        key: key,
        bubbles: true
    })
    el.dispatchEvent(evtDown)
    let evtPress = new KeyboardEvent('keypress',
    {
        keyCode: keyCode,
        which: keyCode,
        code: key,
        key: key,
        bubbles: true
    })
    el.dispatchEvent(evtPress)
}
//__________________________________________________________________________________

/*
───────────────────────────────────────────────────────────────────────────────────────────────────
─██████████████─██████████████─██████──────────██████─██████──██████─██████████████─██████████████─
─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██████████──██░░██─██░░██──██░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─
─██░░██████████─██░░██████░░██─██░░░░░░░░░░██──██░░██─██░░██──██░░██─██░░██████░░██─██░░██████████─
─██░░██─────────██░░██──██░░██─██░░██████░░██──██░░██─██░░██──██░░██─██░░██──██░░██─██░░██─────────
─██░░██─────────██░░██████░░██─██░░██──██░░██──██░░██─██░░██──██░░██─██░░██████░░██─██░░██████████─
─██░░██─────────██░░░░░░░░░░██─██░░██──██░░██──██░░██─██░░██──██░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─
─██░░██─────────██░░██████░░██─██░░██──██░░██──██░░██─██░░██──██░░██─██░░██████░░██─██████████░░██─
─██░░██─────────██░░██──██░░██─██░░██──██░░██████░░██─██░░░░██░░░░██─██░░██──██░░██─────────██░░██─
─██░░██████████─██░░██──██░░██─██░░██──██░░░░░░░░░░██─████░░░░░░████─██░░██──██░░██─██████████░░██─
─██░░░░░░░░░░██─██░░██──██░░██─██░░██──██████████░░██───████░░████───██░░██──██░░██─██░░░░░░░░░░██─
─██████████████─██████──██████─██████──────────██████─────██████─────██████──██████─██████████████─
───────────────────────────────────────────────────────────────────────────────────────────────────
*/
//_________________________________________________________________
//var canvas_solid2 = document.getElementById('myCanvas_solid2');
/*var canvas_movable2 = document.getElementById('myCanvas_movable2')
var canvas_destroyable2 = document.getElementById('myCanvas_destroyable2')
var canvas_dangerous2 = document.getElementById('myCanvas_dangerous2')
var canvas_gettable2 = document.getElementById('myCanvas_gettable2')
var canvas_portal2 = document.getElementById('myCanvas_portal2')
var canvas_usable2 = document.getElementById('myCanvas_usable2')
var canvas_changeable2 = document.getElementById('myCanvas_changeable2')
var canvas_ui2 = document.getElementById('myCanvas_ui2')
var canvas_list2 = [canvas_solid2, canvas_movable2, canvas_destroyable2, canvas_dangerous2, canvas_gettable2, canvas_portal2, canvas_usable2, canvas_changeable2, canvas_ui2];
*/
var canvas_solid = document.getElementById('myCanvas_solid');//set the URL from json later
var canvas_movable = document.getElementById('myCanvas_movable')
var canvas_destroyable = document.getElementById('myCanvas_destroyable')
var canvas_dangerous = document.getElementById('myCanvas_dangerous')
var canvas_gettable = document.getElementById('myCanvas_gettable')
var canvas_portal = document.getElementById('myCanvas_portal')
var canvas_usable = document.getElementById('myCanvas_usable')
var canvas_changeable = document.getElementById('myCanvas_changeable')
var canvas_ui = document.getElementById('myCanvas_ui')
var canvas_list = [canvas_solid, canvas_movable, canvas_destroyable, canvas_dangerous, canvas_gettable, canvas_portal, canvas_usable, canvas_changeable, canvas_ui];
//_________________________________________________________________
/*
──────────────────────────────────────────────────────────────────────────────
─██████████████─██████──██████─██████████████─██████████████─██████──████████─
─██░░░░░░░░░░██─██░░██──██░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██──██░░░░██─
─██░░██████████─██░░██──██░░██─██░░██████████─██░░██████████─██░░██──██░░████─
─██░░██─────────██░░██──██░░██─██░░██─────────██░░██─────────██░░██──██░░██───
─██░░██─────────██░░██████░░██─██░░██████████─██░░██─────────██░░██████░░██───
─██░░██─────────██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██─────────██░░░░░░░░░░██───
─██░░██─────────██░░██████░░██─██░░██████████─██░░██─────────██░░██████░░██───
─██░░██─────────██░░██──██░░██─██░░██─────────██░░██─────────██░░██──██░░██───
─██░░██████████─██░░██──██░░██─██░░██████████─██░░██████████─██░░██──██░░████─
─██░░░░░░░░░░██─██░░██──██░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██──██░░░░██─
─██████████████─██████──██████─██████████████─██████████████─██████──████████─
──────────────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────────────────
─██████████████───██████████████─████████──████████─██████████████─██████████████─
─██░░░░░░░░░░██───██░░░░░░░░░░██─██░░░░██──██░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─
─██░░██████░░██───██░░██████░░██─████░░██──██░░████─██░░██████████─██░░██████████─
─██░░██──██░░██───██░░██──██░░██───██░░░░██░░░░██───██░░██─────────██░░██─────────
─██░░██████░░████─██░░██──██░░██───████░░░░░░████───██░░██████████─██░░██████████─
─██░░░░░░░░░░░░██─██░░██──██░░██─────██░░░░░░██─────██░░░░░░░░░░██─██░░░░░░░░░░██─
─██░░████████░░██─██░░██──██░░██───████░░░░░░████───██░░██████████─██████████░░██─
─██░░██────██░░██─██░░██──██░░██───██░░░░██░░░░██───██░░██─────────────────██░░██─
─██░░████████░░██─██░░██████░░██─████░░██──██░░████─██░░██████████─██████████░░██─
─██░░░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░██──██░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─
─████████████████─██████████████─████████──████████─██████████████─██████████████─
──────────────────────────────────────────────────────────────────────────────────

*/

//__________________________________________________________________________________checkboxes
var checkQ = document.getElementById("cbQ");
checkQ.addEventListener('change', function(e)
{
    if(checkQ.checked)
    {
        simulate(checkQ, 81, "Q");
    }
    if(checkQ.checked == false)
    {

        poses = mydata['output']['tiles']['tile_' + num]['locations'];
        for(i = 0; i < Object.keys(poses).length; i++)
        {
            pos_x = poses['location_' + i]['x'];
            pos_y = poses['location_' + i]['y'];
            draw_b(pos_x, pos_y, canvas_solid, 16, 16);
        }
    }
});

var checkW = document.getElementById("cbW");
checkW.addEventListener('change', function(e)
{
    if(checkW.checked)
    {
        simulate(checkW, 87, "W");
    }
    if(checkW.checked == false)
    {

        poses = mydata['output']['tiles']['tile_' + num]['locations'];
        for(i = 0; i < Object.keys(poses).length; i++)
        {
            pos_x = poses['location_' + i]['x'];
            pos_y = poses['location_' + i]['y'];
            draw_b(pos_x, pos_y, canvas_movable, 16, 16);
        }
    }
});
var checkE = document.getElementById("cbE");
checkE.addEventListener('change', function(e)
{
    if(checkE.checked)
    {
        simulate(checkE, 69, "E");
    }
    if(checkE.checked == false)
    {

        poses = mydata['output']['tiles']['tile_' + num]['locations'];
        for(i = 0; i < Object.keys(poses).length; i++)
        {
            pos_x = poses['location_' + i]['x'];
            pos_y = poses['location_' + i]['y'];
            draw_b(pos_x, pos_y, canvas_destroyable, 16, 16);
        }
    }
});
var checkA = document.getElementById("cbA");
checkA.addEventListener('change', function(e)
{
    if(checkA.checked)
    {
        simulate(checkA, 65, "A");
    }
    if(checkA.checked == false)
    {

        poses = mydata['output']['tiles']['tile_' + num]['locations'];
        for(i = 0; i < Object.keys(poses).length; i++)
        {
            pos_x = poses['location_' + i]['x'];
            pos_y = poses['location_' + i]['y'];
            draw_b(pos_x, pos_y, canvas_dangerous, 16, 16);
        }
    }
});
var checkS = document.getElementById("cbS");
checkS.addEventListener('change', function(e)
{
    if(checkS.checked)
    {
        simulate(checkS, 83, "S");
    }
    if(checkS.checked == false)
    {

        poses = mydata['output']['tiles']['tile_' + num]['locations'];
        for(i = 0; i < Object.keys(poses).length; i++)
        {
            pos_x = poses['location_' + i]['x'];
            pos_y = poses['location_' + i]['y'];
            draw_b(pos_x, pos_y, canvas_gettable, 16, 16);
        }
    }
});
var checkD = document.getElementById("cbD");
checkD.addEventListener('change', function(e)
{
    if(checkD.checked)
    {
        simulate(checkD, 68, "D");
    }
    if(checkD.checked == false)
    {

        poses = mydata['output']['tiles']['tile_' + num]['locations'];
        for(i = 0; i < Object.keys(poses).length; i++)
        {
            pos_x = poses['location_' + i]['x'];
            pos_y = poses['location_' + i]['y'];
            draw_b(pos_x, pos_y, canvas_portal, 16, 16);
        }
    }
});
var checkZ = document.getElementById("cbZ");
checkZ.addEventListener('change', function(e)
{
    if(checkZ.checked)
    {
        simulate(checkZ, 90, "Z");
    }
    if(checkZ.checked == false)
    {

        poses = mydata['output']['tiles']['tile_' + num]['locations'];
        for(i = 0; i < Object.keys(poses).length; i++)
        {
            pos_x = poses['location_' + i]['x'];
            pos_y = poses['location_' + i]['y'];
            draw_b(pos_x, pos_y, canvas_usable, 16, 16);
        }
    }
});
var checkX = document.getElementById("cbX");
checkX.addEventListener('change', function(e)
{
    if(checkX.checked)
    {
        simulate(checkX, 88, "X");
    }
    if(checkX.checked == false)
    {

        poses = mydata['output']['tiles']['tile_' + num]['locations'];
        for(i = 0; i < Object.keys(poses).length; i++)
        {
            pos_x = poses['location_' + i]['x'];
            pos_y = poses['location_' + i]['y'];
            draw_b(pos_x, pos_y, canvas_changeable, 16, 16);
        }
    }
});
var checkC = document.getElementById("cbC");
checkC.addEventListener('change', function(e)
{
    if(checkC.checked)
    {
        simulate(checkC, 67, "C");
    }
    if(checkC.checked == false)
    {

        poses = mydata['output']['tiles']['tile_' + num]['locations'];
        for(i = 0; i < Object.keys(poses).length; i++)
        {
            pos_x = poses['location_' + i]['x'];
            pos_y = poses['location_' + i]['y'];
            draw_b(pos_x, pos_y, canvas_ui, 16, 16);
        }
    }
});

//_____________________________________________________________________________________
/*
─────────────────────────────────────────────────────────────────────
─────────██████─██████████████─██████████████─██████──────────██████─
─────────██░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██████████──██░░██─
─────────██░░██─██░░██████████─██░░██████░░██─██░░░░░░░░░░██──██░░██─
─────────██░░██─██░░██─────────██░░██──██░░██─██░░██████░░██──██░░██─
─────────██░░██─██░░██████████─██░░██──██░░██─██░░██──██░░██──██░░██─
─────────██░░██─██░░░░░░░░░░██─██░░██──██░░██─██░░██──██░░██──██░░██─
─██████──██░░██─██████████░░██─██░░██──██░░██─██░░██──██░░██──██░░██─
─██░░██──██░░██─────────██░░██─██░░██──██░░██─██░░██──██░░██████░░██─
─██░░██████░░██─██████████░░██─██░░██████░░██─██░░██──██░░░░░░░░░░██─
─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██──██████████░░██─
─██████████████─██████████████─██████████████─██████──────────██████─
─────────────────────────────────────────────────────────────────────

*/
//_____________________________________________________________________________________
var mydata;
var num = 0;
var outdata;
var output; //
//_________________________________________

var i;//replace its name
/*
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
─██████████████───██████──██████─██████████████─██████████████─██████████████─██████──────────██████─██████████████─
─██░░░░░░░░░░██───██░░██──██░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██████████──██░░██─██░░░░░░░░░░██─
─██░░██████░░██───██░░██──██░░██─██████░░██████─██████░░██████─██░░██████░░██─██░░░░░░░░░░██──██░░██─██░░██████████─
─██░░██──██░░██───██░░██──██░░██─────██░░██─────────██░░██─────██░░██──██░░██─██░░██████░░██──██░░██─██░░██─────────
─██░░██████░░████─██░░██──██░░██─────██░░██─────────██░░██─────██░░██──██░░██─██░░██──██░░██──██░░██─██░░██████████─
─██░░░░░░░░░░░░██─██░░██──██░░██─────██░░██─────────██░░██─────██░░██──██░░██─██░░██──██░░██──██░░██─██░░░░░░░░░░██─
─██░░████████░░██─██░░██──██░░██─────██░░██─────────██░░██─────██░░██──██░░██─██░░██──██░░██──██░░██─██████████░░██─
─██░░██────██░░██─██░░██──██░░██─────██░░██─────────██░░██─────██░░██──██░░██─██░░██──██░░██████░░██─────────██░░██─
─██░░████████░░██─██░░██████░░██─────██░░██─────────██░░██─────██░░██████░░██─██░░██──██░░░░░░░░░░██─██████████░░██─
─██░░░░░░░░░░░░██─██░░░░░░░░░░██─────██░░██─────────██░░██─────██░░░░░░░░░░██─██░░██──██████████░░██─██░░░░░░░░░░██─
─████████████████─██████████████─────██████─────────██████─────██████████████─██████──────────██████─██████████████─
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

*/

//__________________________________________________________________________________


var name_big = document.getElementById("name_big");


var drawinn = document.getElementById("drawing_container");
var canvas_drawinn = document.getElementById("myCanvas_drawing");
draw_b(0, 0, canvas_drawinn, 808, 712);
drawinn.style.display = "none";



var bClose = document.getElementById("bClose");
bClose.onclick = function()
{
    is_big = 0;
    sizex = 4;
    sizey = 4;
    drawinn.style.display = "none";
    draw_picture(canvas_drawinn.toDataURL(), canvas_list[canvas_drawinn.affordance_id], 256, 224);
    canvas_drawinn.affordance_id = "-1";
};

var is_big = 0;

function enlarge_aff()
{
    is_big = 1;
    simulate(this, 32, "Space");
    document.getElementById("name_big").textContent="Current affordance is: " + this.getAttribute("name");
    sizex = 12;
    sizey = 12;
    drawinn.style.display = "block";
    draw_picture(output["tag_images"][Object.keys(output["tag_images"])[this.getAttribute("affordance_id")]], canvas_drawinn, 768, 672);
    canvas_drawinn.affordance_id = this.getAttribute("affordance_id")
}

var bQ = document.getElementById("bQ");
bQ.onclick = enlarge_aff;
var bW = document.getElementById("bW");
bW.onclick = enlarge_aff;
var bE = document.getElementById("bE");
bE.onclick = enlarge_aff;
var bA = document.getElementById("bA");
bA.onclick = enlarge_aff;
var bS = document.getElementById("bS");
bS.onclick = enlarge_aff;
var bD = document.getElementById("bD");
bD.onclick = enlarge_aff;
var bZ = document.getElementById("bZ");
bZ.onclick = enlarge_aff;
var bX = document.getElementById("bX");
bX.onclick = enlarge_aff;
var bC = document.getElementById("bC");
bC.onclick = enlarge_aff;

var b_reset = document.getElementById("b_reset");
b_reset.style.backgroundColor = "red";
b_reset.style.fontSize = "x-large";
b_reset.style.fontVariant = "small-caps";

b_reset.onclick = function()
{
    simulate(b_reset, 27, "ESC");
    checkQ.checked = false;
    checkW.checked = false;
    checkE.checked = false;
    checkA.checked = false;
    checkS.checked = false;
    checkD.checked = false;
    checkZ.checked = false;
    checkX.checked = false;
    checkC.checked = false;
};

var b_save = document.getElementById("b_save");
b_save.style.backgroundColor = "green";
b_save.style.fontSize = "x-large";
b_save.style.fontVariant = "small-caps";
b_save.onclick = function()
{
    simulate(b_save, 32, "Space");
    alert("Saved!");
};
/*
var check_grid_on = 0;
var b_grid = document.getElementById("b_grid");
b_grid.style.backgroundColor = "gray";
b_grid.style.fontSize = "x-large";
b_grid.style.fontVariant = "small-caps";
b_grid.onclick = function()
{
    if(!check_grid_on)
    {
        for (var canvas_id = 0; canvas_id < canvas_list.length; canvas_id++)
        {
            drawGrid(canvas_list[canvas_id], 256, 224, 16, 'rgb(150, 150, 150)');
        }
        check_grid_on = 1;
    }
};

var b_grid_save = document.getElementById("b_grid_save");
b_grid_save.style.backgroundColor = "darkgreen";
b_grid_save.style.fontSize = "x-large";
b_grid_save.style.fontVariant = "small-caps";
b_grid_save.onclick = function()
{
    grid_checked = 1;
};

var b_grid_reset = document.getElementById("b_grid_reset");
b_grid_reset.style.backgroundColor = "darkred";
b_grid_reset.style.fontSize = "x-large";
b_grid_reset.style.fontVariant = "small-caps";
b_grid_reset.onclick = function()
{
    check_grid_on = 0;
    grid_movex = 0;
    grid_movey = 0;
    for (var canvas_id = 0; canvas_id < canvas_list.length; canvas_id++)
    {
        //erase(canvas_list[canvas_id]);
        draw_b(0, 0, canvas_list[canvas_id], 256, 224);
    }
};

var b_grid_up = document.getElementById("b_grid_up");
b_grid_up.style.backgroundColor = 'rgb(100, 100, 100)';
b_grid_up.style.fontSize = "x-large";
b_grid_up.style.fontVariant = "small-caps";
b_grid_up.onclick = function()
{
    grid_movey--;
    for (var canvas_id = 0; canvas_id < canvas_list.length; canvas_id++)
    {
        erase(canvas_list[canvas_id]);
        draw_b(0, 0, canvas_list[canvas_id], 256, 224);
        drawGrid(canvas_list[canvas_id], 256, 224, 16);
    }

};

var b_grid_down = document.getElementById("b_grid_down");
b_grid_down.style.backgroundColor = "gray";
b_grid_down.style.fontSize = "x-large";
b_grid_down.style.fontVariant = "small-caps";
b_grid_down.onclick = function()
{
    grid_movey++;
    for (var canvas_id = 0; canvas_id < canvas_list.length; canvas_id++)
    {
        erase(canvas_list[canvas_id]);
        draw_b(0, 0, canvas_list[canvas_id], 256, 224);
        drawGrid(canvas_list[canvas_id], 256, 224, 16);
    }

};

var b_grid_left = document.getElementById("b_grid_left");
b_grid_left.style.backgroundColor = "gray";
b_grid_left.style.fontSize = "x-large";
b_grid_left.style.fontVariant = "small-caps";
b_grid_left.onclick = function()
{
    grid_movex--;
    for (var canvas_id = 0; canvas_id < canvas_list.length; canvas_id++)
    {
        erase(canvas_list[canvas_id]);
        draw_b(0, 0, canvas_list[canvas_id], 256, 224);
        drawGrid(canvas_list[canvas_id], 256, 224, 16);
    }

};

var b_grid_right = document.getElementById("b_grid_right");
b_grid_right.style.backgroundColor = "gray";
b_grid_right.style.fontSize = "x-large";
b_grid_right.style.fontVariant = "small-caps";
b_grid_right.onclick = function()
{
    grid_movex++;
    for (var canvas_id = 0; canvas_id < canvas_list.length; canvas_id++)
    {
        erase(canvas_list[canvas_id]);
        draw_b(0, 0, canvas_list[canvas_id], 256, 224);
        drawGrid(canvas_list[canvas_id], 256, 224, 16);
    }

};
*/

function myFunction()
{
    var popup = document.getElementById("myPopup");
    popup.classList.toggle("show");
}
//__________________________________________________________________________________
/*
─────────────────────────────────────────────────────────────
─██████─────────██████████████─██████████████─████████████───
─██░░██─────────██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░████─
─██░░██─────────██░░██████░░██─██░░██████░░██─██░░████░░░░██─
─██░░██─────────██░░██──██░░██─██░░██──██░░██─██░░██──██░░██─
─██░░██─────────██░░██──██░░██─██░░██████░░██─██░░██──██░░██─
─██░░██─────────██░░██──██░░██─██░░░░░░░░░░██─██░░██──██░░██─
─██░░██─────────██░░██──██░░██─██░░██████░░██─██░░██──██░░██─
─██░░██─────────██░░██──██░░██─██░░██──██░░██─██░░██──██░░██─
─██░░██████████─██░░██████░░██─██░░██──██░░██─██░░████░░░░██─
─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██──██░░██─██░░░░░░░░████─
─██████████████─██████████████─██████──██████─████████████───
─────────────────────────────────────────────────────────────
───────────────────────────────────────────────────────────────────────────────────────────
─██████████████─██████████████─████████████████───██████████─██████████████─██████████████─
─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░░░██───██░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─
─██░░██████████─██░░██████████─██░░████████░░██───████░░████─██░░██████░░██─██████░░██████─
─██░░██─────────██░░██─────────██░░██────██░░██─────██░░██───██░░██──██░░██─────██░░██─────
─██░░██████████─██░░██─────────██░░████████░░██─────██░░██───██░░██████░░██─────██░░██─────
─██░░░░░░░░░░██─██░░██─────────██░░░░░░░░░░░░██─────██░░██───██░░░░░░░░░░██─────██░░██─────
─██████████░░██─██░░██─────────██░░██████░░████─────██░░██───██░░██████████─────██░░██─────
─────────██░░██─██░░██─────────██░░██──██░░██───────██░░██───██░░██─────────────██░░██─────
─██████████░░██─██░░██████████─██░░██──██░░██████─████░░████─██░░██─────────────██░░██─────
─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██──██░░░░░░██─██░░░░░░██─██░░██─────────────██░░██─────
─██████████████─██████████████─██████──██████████─██████████─██████─────────────██████─────
───────────────────────────────────────────────────────────────────────────────────────────

*/
//__________________________________________________________________________________load the script

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
            output = //get tagger id from somewhere later (?)
            {
                "tagger_id":"TestTagger",
                "image_id":mydata['output']['image_id']
            };
            for (var index = 0; index < Object.keys(mydata['output']['tiles']).length; index++)
            {
                //console.log('inside for');
                var tile_index = 'tile_' + index;
                var act_tile = mydata['output']['tiles'][tile_index];
                //console.log(act_tile);
                output[tile_index] =
                {
                    "tile_id": act_tile['tile_id'],
                    "solid":0,
                    "movable":0,
                    "destroyable":0,
                    "dangerous":0,
                    "gettable":0,
                    "portal":0,
                    "usable":0,
                    "changeable":0,
                    "ui":0
                };
            }
        });
    };
    document.getElementsByTagName("head")[0].appendChild(script);

})();

//__________________________________________________________________________________

/*
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
─████████████───████████████████───██████████████─██████──────────██████─██████████─██████──────────██████─██████████████─
─██░░░░░░░░████─██░░░░░░░░░░░░██───██░░░░░░░░░░██─██░░██──────────██░░██─██░░░░░░██─██░░██████████──██░░██─██░░░░░░░░░░██─
─██░░████░░░░██─██░░████████░░██───██░░██████░░██─██░░██──────────██░░██─████░░████─██░░░░░░░░░░██──██░░██─██░░██████████─
─██░░██──██░░██─██░░██────██░░██───██░░██──██░░██─██░░██──────────██░░██───██░░██───██░░██████░░██──██░░██─██░░██─────────
─██░░██──██░░██─██░░████████░░██───██░░██████░░██─██░░██──██████──██░░██───██░░██───██░░██──██░░██──██░░██─██░░██─────────
─██░░██──██░░██─██░░░░░░░░░░░░██───██░░░░░░░░░░██─██░░██──██░░██──██░░██───██░░██───██░░██──██░░██──██░░██─██░░██──██████─
─██░░██──██░░██─██░░██████░░████───██░░██████░░██─██░░██──██░░██──██░░██───██░░██───██░░██──██░░██──██░░██─██░░██──██░░██─
─██░░██──██░░██─██░░██──██░░██─────██░░██──██░░██─██░░██████░░██████░░██───██░░██───██░░██──██░░██████░░██─██░░██──██░░██─
─██░░████░░░░██─██░░██──██░░██████─██░░██──██░░██─██░░░░░░░░░░░░░░░░░░██─████░░████─██░░██──██░░░░░░░░░░██─██░░██████░░██─
─██░░░░░░░░████─██░░██──██░░░░░░██─██░░██──██░░██─██░░██████░░██████░░██─██░░░░░░██─██░░██──██████████░░██─██░░░░░░░░░░██─
─████████████───██████──██████████─██████──██████─██████──██████──██████─██████████─██████──────────██████─██████████████─
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
─────────────────────────────────────────────────────────────────────
─██████████████─██████████████─██████──────────██████─██████████████─
─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██████████──██░░██─██░░░░░░░░░░██─
─██░░██████████─██░░██████████─██░░░░░░░░░░██──██░░██─██░░██████████─
─██░░██─────────██░░██─────────██░░██████░░██──██░░██─██░░██─────────
─██░░██████████─██░░██─────────██░░██──██░░██──██░░██─██░░██████████─
─██░░░░░░░░░░██─██░░██─────────██░░██──██░░██──██░░██─██░░░░░░░░░░██─
─██░░██████████─██░░██─────────██░░██──██░░██──██░░██─██████████░░██─
─██░░██─────────██░░██─────────██░░██──██░░██████░░██─────────██░░██─
─██░░██─────────██░░██████████─██░░██──██░░░░░░░░░░██─██████████░░██─
─██░░██─────────██░░░░░░░░░░██─██░░██──██████████░░██─██░░░░░░░░░░██─
─██████─────────██████████████─██████──────────██████─██████████████─
─────────────────────────────────────────────────────────────────────
*/
//_______________________________________________________________ different drawing fcns

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

function draw_picture(x, z, size_x, size_y)
{
    if (z.getContext)
    {
        var ctx = z.getContext('2d');
        var img = new Image();
        //load image first ==v
        img.onload = function()
        {
            ctx.drawImage(img, 0, 0, size_x, size_y);
        };
        img.src = x;
    }
    else
    {
        console.log("canvas not found");
    }
}

for (var canvas_id = 0; canvas_id < canvas_list.length; canvas_id++)
{
    draw_b(0, 0, canvas_list[canvas_id], 256, 224);
}
//__________________________________________________________________________________
/*
─────────────────────────────────────────────────────────────
─██████████████─████████████████───██████████─████████████───
─██░░░░░░░░░░██─██░░░░░░░░░░░░██───██░░░░░░██─██░░░░░░░░████─
─██░░██████████─██░░████████░░██───████░░████─██░░████░░░░██─
─██░░██─────────██░░██────██░░██─────██░░██───██░░██──██░░██─
─██░░██─────────██░░████████░░██─────██░░██───██░░██──██░░██─
─██░░██──██████─██░░░░░░░░░░░░██─────██░░██───██░░██──██░░██─
─██░░██──██░░██─██░░██████░░████─────██░░██───██░░██──██░░██─
─██░░██──██░░██─██░░██──██░░██───────██░░██───██░░██──██░░██─
─██░░██████░░██─██░░██──██░░██████─████░░████─██░░████░░░░██─
─██░░░░░░░░░░██─██░░██──██░░░░░░██─██░░░░░░██─██░░░░░░░░████─
─██████████████─██████──██████████─██████████─████████████───
─────────────────────────────────────────────────────────────

*/
/*var drawGrid = function(w, h, id) {
    var canvas = document.getElementById(id);
    var ctx = canvas.getContext('2d');
    ctx.canvas.width  = w;
    ctx.canvas.height = h;

    var data = '<svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg"> \
        <defs> \
            <pattern id="smallGrid" width="4" height="4" patternUnits="userSpaceOnUse"> \
                <path d="M 8 0 L 0 0 0 8" fill="none" stroke="gray" stroke-width="0.5" /> \
            </pattern> \
            <pattern id="grid" width="80" height="80" patternUnits="userSpaceOnUse"> \
                <rect width="80" height="80" fill="url(#smallGrid)" /> \
                <path d="M 80 0 L 0 0 0 80" fill="none" stroke="gray" stroke-width="1" /> \
            </pattern> \
        </defs> \
        <rect width="100%" height="100%" fill="url(#smallGrid)" /> \
    </svg>';

    var DOMURL = window.URL || window.webkitURL || window;

    var img = new Image();
    var svg = new Blob([data], {type: 'image/svg+xml;charset=utf-8'});
    var url = DOMURL.createObjectURL(svg);

    img.onload = function () {
      ctx.drawImage(img, 0, 0);
      DOMURL.revokeObjectURL(url);
    }
    img.src = url;
}*/

/*drawGrid(256, 224, "myCanvas_solid");
drawGrid(256, 224, "myCanvas_movable");
drawGrid(256, 224, "myCanvas_destroyable");
drawGrid(256, 224, "myCanvas_dangerous");
drawGrid(256, 224, "myCanvas_gettable");
drawGrid(256, 224, "myCanvas_portal");
drawGrid(256, 224, "myCanvas_usable");
drawGrid(256, 224, "myCanvas_changeable");
drawGrid(256, 224, "myCanvas_ui");*/


var grid_movex = 0;
var grid_movey = 0;
var grid_checked = 0;


var drawGrid = function(canvas, w, h, step, grid_color)
{
    var ctx = canvas.getContext('2d');
    ctx.beginPath();
    for (var x = -65536 + grid_movex; x <= w; x += step)
    {
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
    }
    // set the color of the line
    ctx.strokeStyle = grid_color;
    ctx.lineWidth = 0.5;
    ctx.globalCompositeOperation='destination-out';
    ctx.globalCompositeOperation='source-over';
    // the stroke will actually paint the current path
    ctx.stroke();
    // for the sake of the example 2nd path
    ctx.beginPath();
    for (var y = -65536 + grid_movey; y <= h; y += step)
    {
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
    }
    // for your original question - you need to stroke only once
    ctx.stroke();
};
for (var canvas_id = 0; canvas_id < canvas_list.length; canvas_id++)
{
    //drawGrid(canvas_list[canvas_id], 256, 224, 16);
}

//drawGrid(canvas_solid2, 256, 224, 16);

//__________________________________________________________________________________
/*
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
─████████████───████████████████───██████████████─██████──────────██████──────────██████████████─██████──────────██████─
─██░░░░░░░░████─██░░░░░░░░░░░░██───██░░░░░░░░░░██─██░░██──────────██░░██──────────██░░░░░░░░░░██─██░░██████████──██░░██─
─██░░████░░░░██─██░░████████░░██───██░░██████░░██─██░░██──────────██░░██──────────██░░██████░░██─██░░░░░░░░░░██──██░░██─
─██░░██──██░░██─██░░██────██░░██───██░░██──██░░██─██░░██──────────██░░██──────────██░░██──██░░██─██░░██████░░██──██░░██─
─██░░██──██░░██─██░░████████░░██───██░░██████░░██─██░░██──██████──██░░██──────────██░░██──██░░██─██░░██──██░░██──██░░██─
─██░░██──██░░██─██░░░░░░░░░░░░██───██░░░░░░░░░░██─██░░██──██░░██──██░░██──────────██░░██──██░░██─██░░██──██░░██──██░░██─
─██░░██──██░░██─██░░██████░░████───██░░██████░░██─██░░██──██░░██──██░░██──────────██░░██──██░░██─██░░██──██░░██──██░░██─
─██░░██──██░░██─██░░██──██░░██─────██░░██──██░░██─██░░██████░░██████░░██──────────██░░██──██░░██─██░░██──██░░██████░░██─
─██░░████░░░░██─██░░██──██░░██████─██░░██──██░░██─██░░░░░░░░░░░░░░░░░░██──────────██░░██████░░██─██░░██──██░░░░░░░░░░██─
─██░░░░░░░░████─██░░██──██░░░░░░██─██░░██──██░░██─██░░██████░░██████░░██──────────██░░░░░░░░░░██─██░░██──██████████░░██─
─████████████───██████──██████████─██████──██████─██████──██████──██████──────────██████████████─██████──────────██████─
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
───────────────────────────────────────────────────────────────────────────────────────────────────────
─██████████████─██████████████─████████████████───██████████████─██████████████─██████──────────██████─
─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░░░██───██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██████████──██░░██─
─██░░██████████─██░░██████████─██░░████████░░██───██░░██████████─██░░██████████─██░░░░░░░░░░██──██░░██─
─██░░██─────────██░░██─────────██░░██────██░░██───██░░██─────────██░░██─────────██░░██████░░██──██░░██─
─██░░██████████─██░░██─────────██░░████████░░██───██░░██████████─██░░██████████─██░░██──██░░██──██░░██─
─██░░░░░░░░░░██─██░░██─────────██░░░░░░░░░░░░██───██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██──██░░██──██░░██─
─██████████░░██─██░░██─────────██░░██████░░████───██░░██████████─██░░██████████─██░░██──██░░██──██░░██─
─────────██░░██─██░░██─────────██░░██──██░░██─────██░░██─────────██░░██─────────██░░██──██░░██████░░██─
─██████████░░██─██░░██████████─██░░██──██░░██████─██░░██████████─██░░██████████─██░░██──██░░░░░░░░░░██─
─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██──██░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██──██████████░░██─
─██████████████─██████████████─██████──██████████─██████████████─██████████████─██████──────────██████─
───────────────────────────────────────────────────────────────────────────────────────────────────────
*/

//__________________________________________________________________________________good drawing code

var shift_down = 0;
var mouseDown = 0;
var canvas_color = 'white';
var sizex = 4;
var sizey = 4;


function draw_mouse_affordances(tmp, canvas, e)
{
    var context = canvas.getContext("2d");
    var pos = getMousePos(canvas, e);
    //var canvas_check = document.elementFromPoint(pos.x, pos.y);

    if(mouseDown)
    {
        context.fillStyle = color;
    	context.fillRect(pos.x - ((pos.x - grid_movex) % sizex), pos.y - ((pos.y - grid_movey) % sizey), sizex, sizey);
    }
}

for (var canvas_id = 0; canvas_id < canvas_list.length; canvas_id++)
{
    canvas_list[canvas_id].onmousedown = function()
    {
        event.preventDefault();
        if(event.button == 0)
        {
            ++mouseDown;
            color = "white";
        }
        if(event.button == 2)
        {
            ++mouseDown;
            color = "black";
        }
    }
    canvas_list[canvas_id].onmouseup = function()
    {
        event.preventDefault();
        if(event.button == 0)
        {

            --mouseDown;
            color = "white";
        }
        if(event.button == 2)
        {

            --mouseDown;
            color = "black";
        }
    }
    window.addEventListener('mousemove', draw_mouse_affordances.bind(null, event, canvas_list[canvas_id]), false);
    window.addEventListener('mousedown', draw_mouse_affordances.bind(null, event, canvas_list[canvas_id]), false);
    canvas_list[canvas_id].addEventListener('contextmenu', event => event.preventDefault());//block rightclick on 'em
}
//----------------------------------------------------------------------------
canvas_drawinn.onmousedown = function()
{
    event.preventDefault();
    if(event.button == 0)
    {
        ++mouseDown;
        color = "white";
    }
    if(event.button == 2)
    {
        ++mouseDown;
        color = "black";
    }
}
canvas_drawinn.onmouseup = function()
{
    event.preventDefault();
    if(event.button == 0)
    {

        --mouseDown;
        color = "white";
    }
    if(event.button == 2)
    {

        --mouseDown;
        color = "black";
    }
}
window.addEventListener('mousemove', draw_mouse_affordances.bind(null, event, canvas_drawinn), false);
window.addEventListener('mousedown', draw_mouse_affordances.bind(null, event, canvas_drawinn), false);
canvas_drawinn.addEventListener('contextmenu', event => event.preventDefault());//block rightclick on 'em
//----------------------------------------------------------------------------








function getMousePos(canvas, evt)
{
    var rect = canvas.getBoundingClientRect();
    return {
        x: (evt.clientX - rect.left) / (rect.right - rect.left) * canvas.width,
        y: (evt.clientY - rect.top) / (rect.bottom - rect.top) * canvas.height
    };
};

//__________________________________________________________________________________
/*
────────────────────────────────────────────────────────────────
─██████─────────██████████████─██████████████─████████████──────
─██░░██─────────██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░████────
─██░░██─────────██░░██████░░██─██░░██████░░██─██░░████░░░░██────
─██░░██─────────██░░██──██░░██─██░░██──██░░██─██░░██──██░░██────
─██░░██─────────██░░██──██░░██─██░░██████░░██─██░░██──██░░██────
─██░░██─────────██░░██──██░░██─██░░░░░░░░░░██─██░░██──██░░██────
─██░░██─────────██░░██──██░░██─██░░██████░░██─██░░██──██░░██────
─██░░██─────────██░░██──██░░██─██░░██──██░░██─██░░██──██░░██────
─██░░██████████─██░░██████░░██─██░░██──██░░██─██░░████░░░░██────
─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██──██░░██─██░░░░░░░░████────
─██████████████─██████████████─██████──██████─████████████──────
────────────────────────────────────────────────────────────────
─────────────────────────────────────────────────────────────
─██████████████─██████████████─██████████████─██████████████─
─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─
─██░░██████░░██─██░░██████░░██─██░░██████████─██░░██████████─
─██░░██──██░░██─██░░██──██░░██─██░░██─────────██░░██─────────
─██░░██████░░██─██░░██████░░██─██░░██─────────██░░██████████─
─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██──██████─██░░░░░░░░░░██─
─██░░██████████─██░░██████░░██─██░░██──██░░██─██░░██████████─
─██░░██─────────██░░██──██░░██─██░░██──██░░██─██░░██─────────
─██░░██─────────██░░██──██░░██─██░░██████░░██─██░░██████████─
─██░░██─────────██░░██──██░░██─██░░░░░░░░░░██─██░░░░░░░░░░██─
─██████─────────██████──██████─██████████████─██████████████─
─────────────────────────────────────────────────────────────

*/
//__________________________________________________________________________________load the stuff

window.addEventListener('load', function()
{
    var drawinn_img = document.getElementById('drawer');
    drawinn_img.src = mydata['output']['image'];
    var solid_img = document.getElementById('solid');
    solid_img.src = mydata['output']['image'];
    var mov_img = document.getElementById('movable');
    mov_img.src = mydata['output']['image'];
    var dest_img = document.getElementById('destroyable');
    dest_img.src = mydata['output']['image'];
    var dang_img = document.getElementById('dangerous');
    dang_img.src = mydata['output']['image'];
    var get_img = document.getElementById('gettable');
    get_img.src = mydata['output']['image'];
    var port_img = document.getElementById('portal');
    port_img.src = mydata['output']['image'];
    var us_img = document.getElementById('usable');
    us_img.src = mydata['output']['image'];
    var chang_img = document.getElementById('changeable');
    chang_img.src = mydata['output']['image'];
    var ui_img = document.getElementById('ui');
    ui_img.src = mydata['output']['image'];
    var scrn_img = document.getElementById('screenshot_preview');
    scrn_img.src = mydata['output']['image'];
    var tile_tmp = document.getElementById('tile');
    tile_tmp.src = mydata['output']['tiles']['tile_0']['tile_data'];
    var poses = mydata['output']['tiles']['tile_' + num]['locations'];
    for(i = 0; i < Object.keys(poses).length; i++)
    {
        pos_x = poses['location_' + i]['x'];
        pos_y = poses['location_' + i]['y'];
        draw(pos_x, pos_y, canvas_draw);
    }
})
//__________________________________________________________________________________
/*
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
─██████████████─██████─────────██████────────────██████──████████─██████████████─████████──████████─██████████████─
─██░░░░░░░░░░██─██░░██─────────██░░██────────────██░░██──██░░░░██─██░░░░░░░░░░██─██░░░░██──██░░░░██─██░░░░░░░░░░██─
─██░░██████░░██─██░░██─────────██░░██────────────██░░██──██░░████─██░░██████████─████░░██──██░░████─██░░██████████─
─██░░██──██░░██─██░░██─────────██░░██────────────██░░██──██░░██───██░░██───────────██░░░░██░░░░██───██░░██─────────
─██░░██████░░██─██░░██─────────██░░██────────────██░░██████░░██───██░░██████████───████░░░░░░████───██░░██████████─
─██░░░░░░░░░░██─██░░██─────────██░░██────────────██░░░░░░░░░░██───██░░░░░░░░░░██─────████░░████─────██░░░░░░░░░░██─
─██░░██████░░██─██░░██─────────██░░██────────────██░░██████░░██───██░░██████████───────██░░██───────██████████░░██─
─██░░██──██░░██─██░░██─────────██░░██────────────██░░██──██░░██───██░░██───────────────██░░██───────────────██░░██─
─██░░██──██░░██─██░░██████████─██░░██████████────██░░██──██░░████─██░░██████████───────██░░██───────██████████░░██─
─██░░██──██░░██─██░░░░░░░░░░██─██░░░░░░░░░░██────██░░██──██░░░░██─██░░░░░░░░░░██───────██░░██───────██░░░░░░░░░░██─
─██████──██████─██████████████─██████████████────██████──████████─██████████████───────██████───────██████████████─
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────
─██████████████─██████████████─████████████████───██████████████────
─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░░░░░░░░░░░██───██░░░░░░░░░░██────
─██░░██████████─██░░██████░░██─██░░████████░░██───██░░██████████────
─██░░██─────────██░░██──██░░██─██░░██────██░░██───██░░██────────────
─██░░██─────────██░░██──██░░██─██░░████████░░██───██░░██████████────
─██░░██─────────██░░██──██░░██─██░░░░░░░░░░░░██───██░░░░░░░░░░██────
─██░░██─────────██░░██──██░░██─██░░██████░░████───██░░██████████────
─██░░██─────────██░░██──██░░██─██░░██──██░░██─────██░░██────────────
─██░░██████████─██░░██████░░██─██░░██──██░░██████─██░░██████████────
─██░░░░░░░░░░██─██░░░░░░░░░░██─██░░██──██░░░░░░██─██░░░░░░░░░░██────
─██████████████─██████████████─██████──██████████─██████████████────
────────────────────────────────────────────────────────────────────

*/
//__________________________________________________________________________________actual keys
/*document.onkeyup = function(event)
{
    switch (event.keyCode)
    {
        case 16: //do this if want shift
            shift_down = 0;
        break;
    }
}*/
var check_grid = 0; //set to 1 later
document.onkeydown = function(event)
{
    var pos_x = 0;
    var pos_y = 0;
    var tiles = mydata['output']['tiles'];
    var poses = mydata['output']['tiles']['tile_' + num]['locations'];
    switch (event.keyCode)
    {
        case 8: //Backspace
                erase(canvas_draw);
                if(check_grid)
                {
                    for(var canvas_id = 0; canvas_id < canvas_list.length; canvas_id++)
                    {
                        drawGrid(canvas_list[canvas_id], 256, 224, 16, 'rgb(0, 0, 0)');
                        drawGrid(canvas_list[canvas_id], 256, 224, 16, 'rgb(0, 0, 0)');
                        drawGrid(canvas_list[canvas_id], 256, 224, 16, 'rgb(150, 150, 150)');
                    }
                }

                num = num - 1;
                if (num == -1)//check evr time that num doesnt exceed amount of tiles
                {
                    num = num + 1;
                    alert('Out of tiles. Current tile num = ' + num);
                }

                tile.src = tiles['tile_' + num]['tile_data'];
                poses = mydata['output']['tiles']['tile_' + num]['locations'];

                for(i = 0; i < Object.keys(poses).length; i++)
                {
                    pos_x = poses['location_' + i]['x'];
                    pos_y = poses['location_' + i]['y'];
                    draw(pos_x, pos_y, canvas_draw);
                }
                //checkboxes_____________________________________________________
                checkQ.checked = false;
                checkW.checked = false;
                checkE.checked = false;
                checkA.checked = false;
                checkS.checked = false;
                checkD.checked = false;
                checkZ.checked = false;
                checkX.checked = false;
                checkC.checked = false;
                //
                output["tag_images"] =
                {
                    "solid": canvas_solid.toDataURL(),
                    "movable":canvas_movable.toDataURL(),
                    "destroyable":canvas_destroyable.toDataURL(),
                    "dangerous":canvas_dangerous.toDataURL(),
                    "gettable":canvas_gettable.toDataURL(),
                    "portal":canvas_portal.toDataURL(),
                    "usable":canvas_usable.toDataURL(),
                    "changeable":canvas_changeable.toDataURL(),
                    "ui":canvas_ui.toDataURL()
                }
        break;
        case 13: //ENTER


            //if(grid_checked)
            //{
                erase(canvas_draw);
                if(check_grid)
                {
                    for(var canvas_id = 0; canvas_id < canvas_list.length; canvas_id++)
                    {
                        drawGrid(canvas_list[canvas_id], 256, 224, 16, 'rgb(0, 0, 0)');
                        drawGrid(canvas_list[canvas_id], 256, 224, 16, 'rgb(0, 0, 0)');
                        drawGrid(canvas_list[canvas_id], 256, 224, 16, 'rgb(150, 150, 150)');
                    }
                }

                num = num + 1;
                if (num == (Object.keys(tiles).length))//check evr time that num doesnt exceed amount of tiles
                {
                    num = num - 1;
                    alert('Out of tiles. Last tile num = ' + num);
                }

                tile.src = tiles['tile_' + num]['tile_data'];
                poses = mydata['output']['tiles']['tile_' + num]['locations'];

                for(i = 0; i < Object.keys(poses).length; i++)
                {
                    pos_x = poses['location_' + i]['x'];
                    pos_y = poses['location_' + i]['y'];
                    draw(pos_x, pos_y, canvas_draw);
                }
                //checkboxes_____________________________________________________
                checkQ.checked = false;
                checkW.checked = false;
                checkE.checked = false;
                checkA.checked = false;
                checkS.checked = false;
                checkD.checked = false;
                checkZ.checked = false;
                checkX.checked = false;
                checkC.checked = false;
                //
                output["tag_images"] =
                {
                    "solid": canvas_solid.toDataURL(),
                    "movable":canvas_movable.toDataURL(),
                    "destroyable":canvas_destroyable.toDataURL(),
                    "dangerous":canvas_dangerous.toDataURL(),
                    "gettable":canvas_gettable.toDataURL(),
                    "portal":canvas_portal.toDataURL(),
                    "usable":canvas_usable.toDataURL(),
                    "changeable":canvas_changeable.toDataURL(),
                    "ui":canvas_ui.toDataURL()
                }
            /*}
            else
            {
                alert("align the grid first!");
            }*/
            //__________________________________________________________
        break;
        case 32: //space to save
            output["tag_images"] =
            {
                "solid": canvas_solid.toDataURL(),
                "movable":canvas_movable.toDataURL(),
                "destroyable":canvas_destroyable.toDataURL(),
                "dangerous":canvas_dangerous.toDataURL(),
                "gettable":canvas_gettable.toDataURL(),
                "portal":canvas_portal.toDataURL(),
                "usable":canvas_usable.toDataURL(),
                "changeable":canvas_changeable.toDataURL(),
                "ui":canvas_ui.toDataURL()
            }
        break;
        //____________________vvv keypress to draw on affordances squares vvv
        case 81: //q

            poses = mydata['output']['tiles']['tile_' + num]['locations'];
            for(i = 0; i < Object.keys(poses).length; i++)
            {
                pos_x = poses['location_' + i]['x'];
                pos_y = poses['location_' + i]['y'];
                draw(pos_x, pos_y, canvas_solid);
            }
            output['tile_'+num]['solid'] = 1;//do for all affordances!!!!!!!!!!!!!
            checkQ.checked = true;
            //also replace all tmp with num
        break;

        case 87: //w

            poses = mydata['output']['tiles']['tile_' + num]['locations'];
            for(i = 0; i < Object.keys(poses).length; i++)
            {
                pos_x = poses['location_' + i]['x'];
                pos_y = poses['location_' + i]['y'];
                draw(pos_x, pos_y, canvas_movable);
            }
            output['tile_'+num]['movable'] = 1;
            checkW.checked = true;
        break;

        case 69: //e
            poses = mydata['output']['tiles']['tile_' + num]['locations'];
            for(i = 0; i < Object.keys(poses).length; i++)
            {
                pos_x = poses['location_' + i]['x'];
                pos_y = poses['location_' + i]['y'];
                draw(pos_x, pos_y, canvas_destroyable);
            }
            output['tile_'+num]['destroyable'] = 1;
            checkE.checked = true;
        break;
        case 65: //a
            poses = mydata['output']['tiles']['tile_' + num]['locations'];
            for(i = 0; i < Object.keys(poses).length; i++)
            {
                pos_x = poses['location_' + i]['x'];
                pos_y = poses['location_' + i]['y'];
                draw(pos_x, pos_y, canvas_dangerous);
            }
            output['tile_'+num]['dangerous'] = 1;
            checkA.checked = true;
        break;
        case 83: //s
            poses = mydata['output']['tiles']['tile_' + num]['locations'];
            for(i = 0; i < Object.keys(poses).length; i++)
            {
                pos_x = poses['location_' + i]['x'];
                pos_y = poses['location_' + i]['y'];
                draw(pos_x, pos_y, canvas_gettable);
            }
            output['tile_'+num]['gettable'] = 1;
            checkS.checked = true;
        break;
        case 68: //d
            poses = mydata['output']['tiles']['tile_' + num]['locations'];
            for(i = 0; i < Object.keys(poses).length; i++)
            {
                pos_x = poses['location_' + i]['x'];
                pos_y = poses['location_' + i]['y'];
                draw(pos_x, pos_y, canvas_portal);
            }
            output['tile_'+num]['portal'] = 1;
            checkD.checked = true;
        break;

        case 90: //z
            poses = mydata['output']['tiles']['tile_' + num]['locations'];
            for(i = 0; i < Object.keys(poses).length; i++)
            {
                pos_x = poses['location_' + i]['x'];
                pos_y = poses['location_' + i]['y'];
                draw(pos_x, pos_y, canvas_usable);
            }
            output['tile_'+num]['usable'] = 1;
            checkZ.checked = true;
        break;

        case 88: //x
            poses = mydata['output']['tiles']['tile_' + num]['locations'];
            for(i = 0; i < Object.keys(poses).length; i++)
            {
                pos_x = poses['location_' + i]['x'];
                pos_y = poses['location_' + i]['y'];
                draw(pos_x, pos_y, canvas_changeable);
            }
            output['tile_'+num]['changeable'] = 1;
            checkX.checked = true;
        break;

        case 67: // c
            poses = mydata['output']['tiles']['tile_' + num]['locations'];
            for(i = 0; i < Object.keys(poses).length; i++)
            {
                pos_x = poses['location_' + i]['x'];
                pos_y = poses['location_' + i]['y'];
                draw(pos_x, pos_y, canvas_ui);
            }
            output['tile_'+num]['ui'] = 1;
            checkC.checked = true;
        break;

        case 27: // ECS
            if(!is_big)
            {


                poses = mydata['output']['tiles']['tile_' + num]['locations'];
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
                    output['tile_'+num]['solid'] = 0;
                    output['tile_'+num]['movable'] = 0;
                    output['tile_'+num]['destroyable'] = 0;
                    output['tile_'+num]['dangerous'] = 0;
                    output['tile_'+num]['gettable'] = 0;
                    output['tile_'+num]['portal'] = 0;
                    output['tile_'+num]['usable'] = 0;
                    output['tile_'+num]['changeable'] = 0;
                    output['tile_'+num]['ui'] = 0;
                    checkQ.checked = false;
                    checkW.checked = false;
                    checkE.checked = false;
                    checkA.checked = false;
                    checkS.checked = false;
                    checkD.checked = false;
                    checkZ.checked = false;
                    checkX.checked = false;
                    checkC.checked = false;
                }
                //save and load current state of affordances here
                //var solid_img = document.getElementById('solid');
                if(num == 0)
                {
                    for (var canvas_id = 0; canvas_id < canvas_list.length; canvas_id++)
                    {
                        draw_b(0, 0, canvas_list[canvas_id], 256, 224);
                    }
                }
                else
                {
                    for (var canvas_id = 0; canvas_id < canvas_list.length; canvas_id++)
                    {
                        draw_picture(output["tag_images"][Object.keys(output["tag_images"])[canvas_id]], canvas_list[canvas_id], 256, 224);
                    }
                }



            }
            else
            {
                alert("Close the big image first!");
            }
        break;

        /*case 16: do this if want shift
            shift_down = 1;
        break;*/
    }
}