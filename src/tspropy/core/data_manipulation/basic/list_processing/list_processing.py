def is_sublist(sub: list, main: list) -> bool:
    """Check if sub is a sublist of main.

    Args:
        sub (list): The sublist to check.
        main (list): The main list to check against.

    Returns:
        bool: True if sub is a sublist of main, False otherwise.
    """
    return all(elem in main for elem in sub)


def convert_to_flat_list(list_of_lists: list) -> list:
    """Convert a list of lists into a flat list. If it is already flat, return it as is.

    Args:
        list_of_lists (list): The list of lists to flatten.

    Returns:
        list: The flattened list.
    """
    flat_list = []
    for item in list_of_lists:
        if isinstance(item, list):
            flat_list.extend(convert_to_flat_list(item))
        else:
            flat_list.append(item)

    return flat_list
