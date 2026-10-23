# Real-world task instructions

Full-Shot and One-Shot share the same task indices and completion requirements.
The instructions below match the current evaluation metadata.

| Task index | Task | Instruction |
| ---: | --- | --- |
| 0 | Transfer Blocks Using Plate | Place the three colored blocks on the plate one at a time in the order red, green, then blue. Tilt the loaded plate to pour all three blocks into the box. |
| 1 | Enter Keypad Number Sequence | Press the keys in the sequence 2-8-2-6-8, fully releasing each key and returning to the waiting position in front of the keypad between presses. |
| 2 | Set the Table | Take the plate, bowl, fork, and knife out of the crate. Place the plate face up in the lower center of the placemat and the bowl open side up in the upper center, in front of the plate. Place the fork on the left and the knife on the right, with the fork tines and knife tip pointing toward the front. Complete these actions in any order. |
| 3 | Arrange Seasoning Jars | Place the three seasoning jars on the two-tier rack in any order, on either tier and in any positions. |
| 4 | Load Plastic Plates Into Rack | Insert the three plastic plates into the rack in any order and in any slots. |
| 5 | Press Button Five Times | Press the button 5 times, fully releasing it and returning to the waiting position after each press. |
| 6 | Place Pot On Induction Cooker | Place the pot, with the spatula inside, on the induction cooker so that most of the pot is over the cooking surface. Turn the handle toward the camera, move the spatula to the basket, and cover the pot with the lid. |
| 7 | Organize Desktop Workstation | Place the pen in the holder, place the eraser in the open drawer and close it, switch off the lamp, and unplug it. These actions may be performed in any order, except that the lamp must be switched off before it is unplugged. |
| 8 | Organize Books And Pen | Center and align the closed book in the target rectangle. Place the pen along the spine of the open book, close the book, and stack it on the first book. |
| 9 | Arrange Family Tea Set | Place the teacup on the cat-head mat, the teapot on the yellow mat, the lid on the teapot, the tongs in their holder, and the canister on the round mat. Complete these placements in any order, provided the teapot is placed before its lid. |
| 10 | Operate Drawers In Sequence | Open the bottom, middle, and top drawers in that order, then close the top, middle, and bottom drawers in that order. |
| 11 | Sort Blocks in Order | Place the red block on the yellow drawer, the yellow block on the white drawer, and the blue block on the dark-green drawer, in that order. |
| 12 | Fill And Place Pen Holder | Hold the pen holder with its opening upward, insert all four pens one by one in any order, then place the filled holder upright. |
| 13 | Organize Cosmetics Storage Box | Open the upper drawer, cap the lipstick, and place the lipstick and eyebrow pencil inside. Close the drawer once both are inside. Along the way, place the serum and cleansing oil in the storage area above, in either order, before or after closing the drawer. |
| 14 | Strike Drum Five Times | Pick up the drumstick and strike the center of the drumhead 5 times continuously. |
| 15 | Apply Dish Soap Repeatedly | Press the dish-soap bottle 5 times continuously, dispensing soap onto the dishcloth. |
| 16 | Season Meat Skewers Repeatedly | Sprinkle the meat skewers with the seasoning shaker 5 times continuously. |
| 17 | Pack Items For Shipping | Place the wireless mouse, then the wired earphones, in the cardboard box. Cover both items with bubble wrap. |
| 18 | Arrange Flowers and Reed Diffuser | Place the pink, white, and yellow flowers consecutively in the vase, in any order. Place the diffuser bottle on the coaster and insert the reed stick into it, in either order. Either diffuser action may occur before or after the flower sequence, but neither may occur between flower placements. |
| 19 | Prepare Breakfast | Place the pig bun on the left, the yellow steamed bun on the right, and the black steamed bun at the front of the upper tier of the two already assembled, in that order. Add the empty tier on top, place the large bun in its center, then cover the steamer with the lid. |

The keypad sequence is entered once: five individual presses in the order
**2, 8, 2, 6, 8**. Each button/key press includes releasing it and returning
to the waiting position. Holding a key down is not a series of presses.
The drum, dish-soap and seasoning tasks each require five repetitions.

Machine-readable instructions are in [real_world_tasks.json](data/real_world_tasks.json).
Existing evaluation IDs such as `transfer_colored_blocks_using_bowl` and
`sort_colored_blocks_into_boxes` remain stable; the instructions describe the
plate and drawer-top destinations.

## Prepare metadata

Simulation instructions are in [simulation_tasks.json](data/simulation_tasks.json).
For simulation subsets, pass `--definitions docs/data/simulation_tasks.json`
to the same preparation command.

LeRobot v3 stores task text in `meta/tasks.parquet` and each episode metadata
file's `tasks` list. Update both together; frame-level `task_index` values,
episode indices, timestamps, states, actions and video segments stay unchanged.

With PyArrow available in the dataset environment:

```bash
python tools/prepare_lerobot_task_metadata.py data/real-world/one-shot --output prepared/one-shot
python tools/prepare_lerobot_task_metadata.py data/real-world/full-shot --output prepared/full-shot
```

Each output is a metadata overlay, not a standalone dataset. It contains only
the corrected task and episode parquet files at their original relative paths.
The source dataset is unchanged. Publish these metadata files together in a
new dataset revision, retaining the data and video assets.
