
import {Button, ButtonGroup} from "@mui/material";
import {Link} from "react-router-dom";

export default function MenuRouter(){
    return(
        <div className="MenuRouter">
            <ButtonGroup>
                <Link to='/Home'>
                    <Button>Home</Button>
                </Link>
                <Link to='/Tetris'>
                    <Button>Tetris</Button>
                </Link>
                <Link to='/Charts'>
                    <Button>Charts</Button>
                </Link>
            </ButtonGroup>
        </div>
    )
}